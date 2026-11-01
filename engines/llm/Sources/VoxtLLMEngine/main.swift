// voxt-llm-engine: Voxt's local LLM cleanup request as a JSON-lines service.
//
// stdin, one JSON object per line, one request at a time:
//   {"op":"generate","id":..,"system":..,"user":..,"max_tokens":N[,"logprobs":k]}
//   {"op":"exit"}
// stdout, one JSON object per line (logs go to stderr):
//   {"event":"ready",...}
//   {"event":"delta","id":..,"text":..}       per streamed chunk
//   {"event":"done","id":..,"text":..,"tokens":[..],"prompt_tokens":N,"finish_reason":"stop"|"length",...}
//   {"event":"error","id":..,"message":..}

import Foundation
import MLX
import MLXLMCommon

// Protocol lines go to a private copy of stdout; fd 1 points at stderr so that a library
// print() cannot corrupt the stream.
let protocolFD = dup(STDOUT_FILENO)
dup2(STDERR_FILENO, STDOUT_FILENO)
let protocolOut = FileHandle(fileDescriptor: protocolFD, closeOnDealloc: false)

/// JSONSerialization raises on NaN/Infinity; map non-finite numbers to null.
func jsonSafe(_ value: Any) -> Any {
    switch value {
    case let double as Double:
        return double.isFinite ? double : NSNull()
    case let float as Float:
        return float.isFinite ? Double(float) : NSNull()
    case let dictionary as [String: Any]:
        return dictionary.mapValues(jsonSafe)
    case let array as [Any]:
        return array.map(jsonSafe)
    default:
        return value
    }
}

func emit(_ object: [String: Any]) {
    let object = jsonSafe(object)
    guard JSONSerialization.isValidJSONObject(object),
        var data = try? JSONSerialization.data(
            withJSONObject: object, options: [.withoutEscapingSlashes, .sortedKeys])
    else {
        EngineLog.info("cannot encode event: \(object)")
        return
    }
    data.append(0x0A)
    protocolOut.write(data)  // one write(2) per line
}

func emitError(_ id: Any?, _ message: String) {
    var object: [String: Any] = ["event": "error", "message": message]
    if let id { object["id"] = id }
    emit(object)
}

func intValue(_ value: Any?) -> Int? {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
        return nil
    }
    return Double(number.intValue) == number.doubleValue ? number.intValue : nil
}

// MARK: - Arguments

var modelDirectory: URL?
var repo = VoxtModelCatalog.defaultModelRepo
var warmup = true
var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let argument = arguments.next() {
    switch argument {
    case "--model":
        guard let value = arguments.next() else { EngineLog.info("--model needs a path"); exit(2) }
        modelDirectory = URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
    case "--repo":
        guard let value = arguments.next() else { EngineLog.info("--repo needs a value"); exit(2) }
        repo = value
    case "--no-warmup":
        warmup = false
    case "-h", "--help":
        EngineLog.info(
            "usage: voxt-llm-engine --model DIR [--repo HF_REPO_ID] [--no-warmup]\n"
                + "  --model      model directory (mlx-community/Qwen3.5-4B-OptiQ-4bit)\n"
                + "  --repo       repo id, used for Voxt's model-family rules (default \(VoxtModelCatalog.defaultModelRepo))\n"
                + "  --no-warmup  skip the warm-up request")
        exit(0)
    default:
        EngineLog.info("unknown argument: \(argument)")
        exit(2)
    }
}
guard let modelDirectory else {
    EngineLog.info("--model is required")
    exit(2)
}

// MARK: - Load and warm up

let loadStart = Date()
let engine: Engine
do {
    engine = try await Engine.load(modelDirectory: modelDirectory, repo: repo)
} catch {
    EngineLog.info("model load failed: \(error)")
    exit(1)
}
let loadMs = Date().timeIntervalSince(loadStart) * 1000
EngineLog.info("loaded \(modelDirectory.lastPathComponent) in \(Int(loadMs)) ms")

var warmupMs = 0.0
if warmup {
    // One short request through the whole path. The ~1.3k-character system prompt lands in
    // the same prefill bucket as Voxt's real prompt, and the warm-up also traces the compiled
    // verify forwards, so the first real request pays no graph building.
    let warmStart = Date()
    let warmSystem = String(repeating: "你是一个语音转写清理助手，请删除语气词并修正标点符号。", count: 46)
    let warmUser =
        "Process this ASR transcription according to the system instructions.\n"
        + "Return only the final processed text.\nInput:\n嗯，那个，今天下午三点开会。"
    do {
        _ = try await engine.generate(
            system: warmSystem, user: warmUser, maxTokens: 16, logprobsTopK: 0, pretrace: true
        ) { _ in }
    } catch {
        EngineLog.info("warm-up failed: \(error)")
        exit(1)
    }
    warmupMs = Date().timeIntervalSince(warmStart) * 1000
    // Release the warm-up's buffers so the first request does not peak on top of them.
    Memory.clearCache()
}

emit([
    "event": "ready",
    "model": modelDirectory.path,
    "repo": engine.repo,
    "load_ms": loadMs,
    "warmup_ms": warmupMs,
])

// MARK: - Request loop

requestLoop: while let line = readLine(strippingNewline: true) {
    if line.allSatisfy({ $0.isWhitespace }) { continue }
    guard let data = line.data(using: .utf8),
        let request = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        emitError(nil, "invalid JSON line")
        continue
    }
    let id = request["id"]

    switch request["op"] as? String {
    case "exit":
        break requestLoop

    case "generate":
        guard let system = request["system"] as? String,
            let user = request["user"] as? String,
            let maxTokens = intValue(request["max_tokens"]), maxTokens > 0
        else {
            emitError(id, "generate needs string system, string user, positive int max_tokens")
            continue
        }
        let logprobsK = intValue(request["logprobs"]) ?? 0
        guard logprobsK >= 0 else {
            emitError(id, "logprobs must be a non-negative int")
            continue
        }
        do {
            let result = try await engine.generate(
                system: system, user: user, maxTokens: maxTokens, logprobsTopK: logprobsK
            ) { chunk in
                var delta: [String: Any] = ["event": "delta", "text": chunk]
                if let id { delta["id"] = id }
                emit(delta)
            }
            var done: [String: Any] = [
                "event": "done",
                "text": result.text,
                "tokens": result.tokens,
                "prompt_tokens": result.promptTokens,
                "finish_reason": result.finishReason,
                "stop_token": result.stopToken as Any? ?? NSNull(),
                "stats": result.stats,
            ]
            if let id { done["id"] = id }
            if let logprobs = result.logprobs { done["logprobs"] = logprobs }
            emit(done)
        } catch {
            emitError(id, "generate failed: \(error)")
        }
        // Voxt clears the MLX buffer pool after each request; do it after the answer is out.
        Memory.clearCache()

    case let op:
        emitError(id, "unknown op: \(op ?? "<missing>")")
    }
}
exit(0)
