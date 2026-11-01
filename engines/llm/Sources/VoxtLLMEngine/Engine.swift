// Loads the model the way Voxt does and runs one cleanup request the way
// CustomLLMModelManager.runLocalPromptRequest does, plus a cross-request prefix cache.

import CoreGraphics
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

nonisolated enum EngineLog {
    static func info(_ message: String) {
        FileHandle.standardError.write(Data(("[engine] " + message + "\n").utf8))
    }
}

struct GenerateResult {
    var text: String
    var tokens: [Int]
    var stopToken: Int?
    var promptTokens: Int
    var finishReason: String
    var logprobs: [[String: Any]]?
    var stats: [String: Any]
}

final class Engine {
    let container: ModelContainer
    let repo: String
    let family: VoxtModelFamily
    let tokenizer: any MLXLMCommon.Tokenizer

    /// Snapshots of the model state after the system prompt. Requests are sequential.
    private let prefixCache = PrefixCacheStore(limit: 2)
    /// Where the varying user content starts, per system prompt (see `structuralBoundary`).
    private var boundaryBySystem: [String: (boundary: Int, userStart: Int)] = [:]
    /// Tokenized system head per system prompt, so only the varying tail is tokenized per
    /// request. Stored only after the split has been checked against the chat template.
    private var promptHeadBySystem: [String: [Int]] = [:]
    /// First user message seen per system prompt and the length of the instruction prefix it
    /// shares with later ones (see `transcriptStart`).
    private var userRefBySystem: [String: [Int]] = [:]
    private var userPrefixLenBySystem: [String: Int] = [:]

    /// The structural boundary is rounded down to the GDN chunk (16 tokens). An unaligned
    /// split perturbs the last bits of the recurrent state and can flip a repetition-penalty
    /// tie; 16-aligned splits reproduce Voxt's own chunked prefill bit for bit.
    private static let boundaryAlignment = 16

    private init(container: ModelContainer, repo: String, tokenizer: any MLXLMCommon.Tokenizer) {
        self.container = container
        self.repo = repo
        self.tokenizer = tokenizer
        self.family = VoxtModelFamily.resolve(for: repo)
    }

    /// CustomLLMModelManager.container(for:) -> MemoryEfficientModelContainerLoader.load
    static func load(modelDirectory: URL, repo: String) async throws -> Engine {
        let canonicalRepo = VoxtModelCatalog.canonicalModelRepo(repo)
        guard !VoxtModelCatalog.requiresVLMFactory(repo: canonicalRepo) else {
            throw NSError(
                domain: "VoxtLLMEngine", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "vision-language models are not supported: \(canonicalRepo)"])
        }
        let container = try await MemoryEfficientModelContainerLoader.load(
            from: modelDirectory,
            using: LocalTokenizerLoader(),
            requiresVLMFactory: false
        )
        let tokenizer = await container.tokenizer
        return Engine(container: container, repo: canonicalRepo, tokenizer: tokenizer)
    }

    // MARK: - Prompt structure

    /// The input ChatSession builds for this request: system + user, thinking off.
    private func userInput(system: String, user: String) -> UserInput {
        UserInput(
            chat: [.system(system), .user(user)],
            processing: .init(resize: CGSize(width: 512, height: 512)),
            tools: nil,
            additionalContext: VoxtRequestPolicy.thinkingOffAdditionalContext)
    }

    /// Qwen3.5 chat template for [system, user], no tools, thinking off, split at the special
    /// token that opens the user message. `head + tail` is exactly the string the template
    /// renders (it trims each message and appends the generation prompt).
    static func promptHead(system: String) -> String {
        "<|im_start|>system\n" + system.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func promptTail(user: String) -> String {
        "<|im_end|>\n<|im_start|>user\n"
            + user.trimmingCharacters(in: .whitespacesAndNewlines)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// The prompt tokens. The first request for a system prompt goes through the real chat
    /// template and checks that head + tail tokenizes to the same tokens; later requests only
    /// tokenize the tail. The split falls on a special token, so it cannot merge across the
    /// boundary; if the check fails, that system prompt keeps using the full template.
    private func promptTokens(system: String, user: String) async throws -> [Int] {
        let tail = Self.promptTail(user: user)
        if let head = promptHeadBySystem[system] {
            return head + tokenizer.encode(text: tail, addSpecialTokens: false)
        }
        let input = try await container.prepare(input: userInput(system: system, user: user))
        let full = input.text.tokens.asArray(Int32.self).map(Int.init)
        let head = tokenizer.encode(text: Self.promptHead(system: system), addSpecialTokens: false)
        if full == head + tokenizer.encode(text: tail, addSpecialTokens: false) {
            promptHeadBySystem[system] = head
        }
        return full
    }

    /// Token index up to which the prompt does not depend on the transcript: the longest
    /// common token prefix with a sibling prompt whose user message starts with a different
    /// character. It covers the system message and the fixed start of the user message, so its
    /// state can be reused by every request with the same system prompt. `boundary` is that
    /// prefix aligned down to `boundaryAlignment`; `userStart` is the unaligned prefix.
    private func structuralBoundary(system: String, user: String, prompt: [Int]) async throws
        -> (boundary: Int, userStart: Int)
    {
        if let cached = boundaryBySystem[system] {
            return (
                min(cached.boundary, prompt.count - 1),
                min(cached.userStart, prompt.count - 1))
        }
        let sentinel = "\u{1}" + String(user.dropFirst())
        let sibling = try await container.prepare(input: userInput(system: system, user: sentinel))
        let siblingTokens = sibling.text.tokens.asArray(Int32.self).map(Int.init)
        let common = commonPrefixLength(prompt, siblingTokens)
        let aligned = (common / Self.boundaryAlignment) * Self.boundaryAlignment
        boundaryBySystem[system] = (aligned, common)
        return (min(aligned, prompt.count - 1), min(common, prompt.count - 1))
    }

    /// First token of the transcript inside the user message. The user message starts with a
    /// fixed instruction; its length is the running minimum of the common token prefix between
    /// this user message and the first one seen for the same system prompt.
    private func transcriptStart(
        system: String, user: String, userRegionStart: Int, promptCount: Int
    ) -> Int {
        let tokens = tokenizer.encode(
            text: user.trimmingCharacters(in: .whitespacesAndNewlines), addSpecialTokens: false)
        guard let reference = userRefBySystem[system] else {
            userRefBySystem[system] = tokens
            return userRegionStart
        }
        let common = commonPrefixLength(reference, tokens)
        let prefix = min(userPrefixLenBySystem[system] ?? common, common)
        userPrefixLenBySystem[system] = prefix
        return min(userRegionStart + prefix, max(0, promptCount - 1))
    }

    // MARK: - Generate

    /// One cleanup request. `pretrace` (warm-up only) also traces the compiled verify shapes.
    func generate(
        system: String,
        user: String,
        maxTokens: Int,
        logprobsTopK: Int,
        pretrace: Bool = false,
        onDelta: (String) -> Void
    ) async throws -> GenerateResult {
        let requestStart = Date()
        let parameters = VoxtRequestPolicy.generationParameters(
            instructions: system, prompt: user, maxTokens: maxTokens, family: family)

        let promptTokens = try await promptTokens(system: system, user: user)
        let (boundary, userRegionStart) = try await structuralBoundary(
            system: system, user: user, prompt: promptTokens)
        let transcriptStart = transcriptStart(
            system: system, user: user, userRegionStart: userRegionStart,
            promptCount: promptTokens.count)

        let payload = GenerationPayload(
            promptTokens: promptTokens,
            boundary: boundary,
            userRegionStart: userRegionStart,
            transcriptStart: transcriptStart,
            parameters: parameters,
            maxTokens: maxTokens,
            logprobsTopK: logprobsTopK,
            pretrace: pretrace)

        // Run the loop on the container's isolation and stream chunks back while it runs.
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let store = prefixCache
        let generationTask = Task { () -> GenerationOutcome in
            do {
                let outcome = try await container.perform(nonSendable: payload) { context, payload in
                    try runGeneration(context: context, payload: payload, store: store) { text in
                        continuation.yield(text)
                    }
                }
                continuation.finish()
                return outcome
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }

        var firstChunkAt: Date?
        do {
            for try await text in stream where !text.isEmpty {
                if firstChunkAt == nil { firstChunkAt = Date() }
                onDelta(text)
            }
        } catch {
            _ = try? await generationTask.value
            throw error
        }
        let outcome = try await generationTask.value

        let ms = { (date: Date) in date.timeIntervalSince(requestStart) * 1000 }
        var stats: [String: Any] = [
            "total_ms": ms(Date()),
            "prompt_time_s": outcome.prefillSeconds,
            "generation_time_s": outcome.generateSeconds,
            "generation_tokens": outcome.tokens.count,
            "cached_prompt_tokens": outcome.cachedPromptTokens,
            "spec_rounds": outcome.rounds,
            "spec_accepted_drafts": outcome.acceptedDrafts,
            "spec_drafted_tokens": outcome.draftedTokens,
        ]
        if let firstChunkAt { stats["ttft_ms"] = ms(firstChunkAt) }

        return GenerateResult(
            text: outcome.text,
            tokens: outcome.tokens,
            stopToken: outcome.stopToken,
            promptTokens: outcome.promptTokens,
            finishReason: outcome.finishReason,
            logprobs: outcome.logprobs,
            stats: stats)
    }
}
