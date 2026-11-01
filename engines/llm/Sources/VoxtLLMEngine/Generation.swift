// Cross-request prefix cache and the generation loop.
//
// Voxt runs every request through ChatSession with a fresh cache, so its ~950-token system
// prompt is prefilled on every request. Here the model state (attention KV caches and the
// Gated DeltaNet conv/recurrent states) after the part of the prompt that does not depend on
// the transcript is kept across requests, and only the rest is prefilled.
//
// Decoding is Voxt's policy: greedy with the repetition penalty (1.05 over the last 32 tokens,
// seeded with the prompt), the same stop tokens and max_tokens. It is sped up with exact
// prompt-lookup speculative decoding: draft tokens are copied from the transcript and checked
// by one forward of the model; a draft token is kept only if it is the token greedy decoding
// would have picked.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN  // logSoftmax

// MARK: - Prefix snapshots

/// The cache state of every layer after a token prefix, held by value.
///
/// `KVCache.copy()` copies the cache objects (the MLX arrays themselves are immutable and
/// shared), and restoring copies again, so a request can never modify a snapshot.
nonisolated final class PrefixSnapshot: @unchecked Sendable {
    let tokens: [Int]
    private let caches: [any KVCache]

    init(tokens: [Int], caches: [any KVCache]) {
        self.tokens = tokens
        self.caches = caches.map { $0.copy() }
    }

    func restore() -> [any KVCache] {
        caches.map { $0.copy() }
    }
}

/// A few prefix snapshots; `best(for:)` returns the longest one that is a prefix of the prompt.
nonisolated final class PrefixCacheStore: @unchecked Sendable {
    private var snapshots: [PrefixSnapshot] = []
    private let limit: Int
    /// Prompt of the previous request, for the cross-request common-prefix boundary.
    var previousPrompt: [Int]? = nil

    init(limit: Int) {
        self.limit = limit
    }

    func best(for prompt: [Int]) -> PrefixSnapshot? {
        snapshots
            .filter { $0.tokens.count < prompt.count && prompt.starts(with: $0.tokens) }
            .max { $0.tokens.count < $1.tokens.count }
    }

    func add(tokens: [Int], caches: [any KVCache]) {
        snapshots.removeAll { $0.tokens == tokens }
        snapshots.append(PrefixSnapshot(tokens: tokens, caches: caches))
        snapshots.sort { $0.tokens.count < $1.tokens.count }
        if snapshots.count > limit {
            // Keep the two shortest (the system-prompt bases) and the longest others.
            snapshots = Array(snapshots.prefix(2)) + Array(snapshots.suffix(limit - 2))
        }
    }

    func clear() {
        snapshots.removeAll()
        previousPrompt = nil
    }
}

nonisolated func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var n = 0
    let m = min(a.count, b.count)
    while n < m, a[n] == b[n] { n += 1 }
    return n
}

// MARK: - Request and result

nonisolated struct GenerationPayload: Sendable {
    let promptTokens: [Int]
    /// Prompt index up to which the state does not depend on the transcript.
    let boundary: Int
    /// First token of the user message: drafts are looked up from here on.
    let userRegionStart: Int
    /// First token of the transcript inside the user message: the positional draft starts here.
    let transcriptStart: Int
    let parameters: GenerateParameters
    let maxTokens: Int
    let logprobsTopK: Int
    /// Warm-up only: trace the compiled verify forward for every row count the loop uses.
    let pretrace: Bool
}

nonisolated struct GenerationOutcome: @unchecked Sendable {
    var text: String
    var tokens: [Int]
    var stopToken: Int?
    var promptTokens: Int
    var finishReason: String
    var logprobs: [[String: Any]]?
    var prefillSeconds: Double
    var generateSeconds: Double
    var cachedPromptTokens: Int
    var rounds: Int
    var acceptedDrafts: Int
    var draftedTokens: Int
}

// MARK: - Helpers

nonisolated func buildStopTokenIDs(
    configuration: ModelConfiguration, tokenizer: any Tokenizer
) -> Set<Int> {
    var ids = configuration.eosTokenIds
    if let eos = tokenizer.eosTokenId {
        ids.insert(eos)
    }
    for token in configuration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            ids.insert(id)
        }
    }
    return ids
}

/// Feed a token range through the model only to advance the cache, chunked like
/// `LLMModel.prepare` but without leaving a remainder, so the state can be snapshotted
/// exactly at the end of the range.
nonisolated func prefillOnly(
    model: any LanguageModel, cache: [any KVCache], tokens: ArraySlice<Int>, stepSize: Int
) {
    var start = tokens.startIndex
    while start < tokens.endIndex {
        let end = min(start + stepSize, tokens.endIndex)
        let chunk = LMInput.Text(
            tokens: MLXArray(tokens[start ..< end].map { Int32($0) })[.newAxis])
        _ = model(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
        asyncEval(cache)
        start = end
    }
    eval(cache)
}

// MARK: - Generation loop

/// One request. Runs inside `ModelContainer.perform`; `emitChunk` receives text as it is
/// decoded.
nonisolated func runGeneration(
    context: ModelContext,
    payload: GenerationPayload,
    store: PrefixCacheStore,
    emitChunk: (String) -> Void
) throws -> GenerationOutcome {
    let model = context.model
    let qwen = model as? Qwen35Model
    let tokenizer = context.tokenizer
    let prompt = payload.promptTokens
    let params = payload.parameters
    let stepSize = params.prefill.resolvedStepSize()
    let prefillStart = Date()

    // 1. Start from the longest cached prefix of this prompt, or from an empty cache.
    var cache: [any KVCache]
    var start: Int
    if let restored = store.best(for: prompt) {
        cache = restored.restore()
        start = restored.tokens.count
    } else {
        // A different system prompt: the old snapshots will not match again soon.
        store.clear()
        cache = try model.newCache(parameters: params)
        start = 0
    }

    // 2. Prefill up to the reusable boundary and snapshot the state there. The structural
    //    boundary (system message + fixed start of the user message) is always reusable; the
    //    common prefix with the previous request, aligned down to 64 tokens, may be longer.
    var boundary = max(start, min(payload.boundary, prompt.count - 1))
    if let previous = store.previousPrompt {
        let aligned = (commonPrefixLength(prompt, previous) / 64) * 64
        if aligned > boundary, aligned < prompt.count {
            boundary = aligned
        }
    }
    store.previousPrompt = prompt
    if boundary > start {
        prefillOnly(model: model, cache: cache, tokens: prompt[start ..< boundary], stepSize: stepSize)
        store.add(tokens: Array(prompt[0 ..< boundary]), caches: cache)
        start = boundary
    }

    // 2b. The 14 tokens after the structural boundary are the fixed instruction that opens
    //     every user message, so they are prefilled once and snapshotted too. The request's
    //     own suffix then starts 14 rows later, which can save a 32-row matmul tile. Both
    //     forwards pad their attention queries so that every row keeps the geometry of a
    //     forward that started at the boundary: the rows are bit-identical to it.
    let templateLength = 14
    var paddedStart = -1
    if let qwen, boundary == payload.userRegionStart, prompt.count > boundary + templateLength {
        let after = boundary + templateLength
        let tilesBefore = (prompt.count - boundary + 31) / 32
        let tilesAfter = (prompt.count - after + 31) / 32
        if prompt.count - after >= 12, tilesAfter < tilesBefore {
            let template = MLXArray(prompt[boundary ..< after].map { Int32($0) })[.newAxis]
            _ = qwen.prefillPadded(template, cache: cache, padQueries: 23, padKeys: 997)
            eval(cache)
            store.add(tokens: Array(prompt[0 ..< after]), caches: cache)
            start = after
            paddedStart = after
        }
    }

    // 3. Prefill the rest like TokenIterator: chunked with the request's step size, then one
    //    forward of the remainder, whose last row gives the first token.
    let suffixInput = LMInput(tokens: MLXArray(prompt[start...].map { Int32($0) }))
    var state: LMOutput.State? = nil
    let logits: MLXArray
    if let qwen, start == paddedStart {
        logits = withPreparedCache(cache, lengths: suffixInput.text.sequenceLengths) {
            qwen.prefillPadded(
                suffixInput.text.tokens[.newAxis], cache: cache, padQueries: templateLength,
                padKeys: 0)
        }
    } else {
        switch try model.prepare(suffixInput, cache: cache, state: nil, prefill: params.prefill) {
        case .tokens(let remainder):
            // Only the last row's logits are needed (the first token); skip the LM head on the
            // other rows.
            let out = withPreparedCache(cache, lengths: remainder.sequenceLengths) {
                if let qwen {
                    return qwen.callAsFunctionLastRow(
                        remainder[text: .newAxis], cache: cache.isEmpty ? nil : cache)
                }
                return model(remainder[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil)
            }
            state = out.state
            logits = out.logits
        case .logits(let out):
            state = out.state
            logits = out.logits
        }
    }
    let prefillSeconds = Date().timeIntervalSince(prefillStart)

    // 4. Decode.
    //
    // The cache holds every token except the `pending` tail. A verify round feeds
    // `pending + draft` in one forward; row `pending.count - 1 + i` predicts draft[i]. A draft
    // token is accepted if it equals the penalized greedy pick of its row (with the repetition
    // window advanced by each accepted token); the first mismatching row supplies the next
    // token. Drafts are the continuation of the longest suffix match of the context inside the
    // user message and the output so far, or else the transcript from a positional pointer.
    let stopTokenIDs = buildStopTokenIDs(configuration: context.configuration, tokenizer: tokenizer)
    let unknownTokenID = tokenizer.unknownTokenId
    let sampler = params.sampler()
    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    var toolProcessor = ToolCallProcessor(
        format: context.configuration.toolCallFormat ?? .json,
        tools: nil,
        toolCallPolicy: params.toolCallPolicy)

    struct StepLogprobs {
        let indices: MLXArray
        let values: MLXArray
        let chosen: MLXArray
    }

    struct CacheSnapshot {
        struct Recurrent {
            let index: Int
            let conv: MLXArray?
            let rec: MLXArray?
            let offset: Int
        }
        struct Attention {
            let index: Int
            let offset: Int
        }
        var recurrent: [Recurrent] = []
        var attention: [Attention] = []
    }

    let topK = payload.logprobsTopK
    let userRegionStart = payload.userRegionStart
    let maxDraft = 32  // rows per verify forward
    let ngramMax = 3
    let foldCap = 12  // pending tokens folded into the next verify forward before a replay
    // A penalized top-2 margin below this is a near tie: the verify forward's rounding may
    // differ from a single-token step there.
    let marginThreshold: Float = 0.15

    var generated: [Int] = []
    var sampled: [Int] = []
    var steps: [StepLogprobs] = []
    var aggregated = ""
    var finishReason = "length"
    var stopToken: Int? = nil
    var pending: [Int] = []
    var looping = true
    var rounds = 0
    var acceptedDrafts = 0
    var draftedTokens = 0
    // GDN layer inputs saved by the last verify forward (see `truncateCache`).
    var savedGDNInputs: [Int: MLXArray] = [:]

    // Position in the transcript that the output is expected to continue from.
    var positionalPointer = payload.transcriptStart

    /// Advance the positional pointer over emitted tokens: a match moves one step; otherwise
    /// look ahead up to 16 tokens to re-synchronize, else skip one.
    func advancePointer(over tokens: [Int]) {
        for token in tokens {
            if positionalPointer < prompt.count, prompt[positionalPointer] == token {
                positionalPointer += 1
            } else {
                let end = min(prompt.count, positionalPointer + 16)
                if let q = (positionalPointer + 1 ..< max(positionalPointer + 1, end))
                    .first(where: { prompt[$0] == token })
                {
                    positionalPointer = q + 1
                } else {
                    positionalPointer += 1
                }
            }
            positionalPointer = min(positionalPointer, prompt.count)
        }
    }

    func emit(_ text: String) {
        guard !text.isEmpty else { return }
        aggregated += text
        emitChunk(text)
    }

    func emitToken(_ token: Int) {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            for output in toolProcessor.processChunkOutputs(chunk) {
                if case .response(let text) = output {
                    emit(text)
                }
            }
        }
    }

    /// Raw (pre-penalty) logprobs of the rows that produced `chosen`, one row per token. The
    /// arrays stay lazy until the response is serialized.
    func logprobSteps(rows: MLXArray, chosen: [Int]) -> [StepLogprobs] {
        guard topK > 0, !chosen.isEmpty else { return [] }
        let logprobs = logSoftmax(rows.asType(.float32), axis: -1)
        let k = min(topK, logprobs.dim(-1))
        let indices = argPartition(-logprobs, kth: k - 1, axis: -1)[0..., ..<k]
        let values = takeAlong(logprobs, indices, axis: -1)
        let chosenArray = MLXArray(chosen.map { Int32($0) }).reshaped([chosen.count, 1])
        let chosenLP = takeAlong(logprobs, chosenArray, axis: -1)
        asyncEval([indices, values, chosenLP])
        return (0 ..< chosen.count).map { i in
            StepLogprobs(
                indices: indices[i ..< (i + 1)],
                values: values[i ..< (i + 1)],
                chosen: chosenLP[i ..< (i + 1)])
        }
    }

    /// Raw logprobs of one row for a token that is still a lazy array.
    func logprobStep(row: MLXArray, token: MLXArray) -> StepLogprobs? {
        guard topK > 0 else { return nil }
        let logprobs = logSoftmax(row.asType(.float32), axis: -1)
        let k = min(topK, logprobs.dim(-1))
        let indices = argPartition(-logprobs, kth: k - 1, axis: -1)[0..., ..<k]
        let values = takeAlong(logprobs, indices, axis: -1)
        let chosen = takeAlong(logprobs, token.reshaped([1, 1]).asType(.int32), axis: -1)
        asyncEval([indices, values, chosen])
        return StepLogprobs(indices: indices, values: values, chosen: chosen)
    }

    func snapshotCache() -> CacheSnapshot {
        var snap = CacheSnapshot()
        for (i, c) in cache.enumerated() {
            if let m = c as? MambaCache {
                snap.recurrent.append(.init(index: i, conv: m[0], rec: m[1], offset: m.offset))
            } else {
                snap.attention.append(.init(index: i, offset: c.offset))
            }
        }
        return snap
    }

    /// Roll the cache back to a snapshot: recurrent states by reference, KV caches by offset.
    func restoreCache(_ snap: CacheSnapshot) {
        for e in snap.recurrent {
            guard let m = cache[e.index] as? MambaCache else { continue }
            m[0] = e.conv
            m[1] = e.rec
            m.offset = e.offset
        }
        for e in snap.attention {
            let c = cache[e.index]
            let drop = c.offset - e.offset
            if drop > 0 { c.trim(drop) }
        }
    }

    /// One forward over `ids`; returns logits [1, T, V] and advances the cache. A T > 1
    /// forward runs through the model's compiled layer schedule (same logits as the general
    /// path, about 8% faster at T = 32).
    func forward(_ ids: [Int]) -> MLXArray {
        let tokens = MLXArray(ids.map { Int32($0) })
        if ids.count > 1, state == nil, let qwen {
            return qwen.verifyCompiled(tokens[.newAxis], cache: cache)
        }
        let out = model(LMInput.Text(tokens: tokens)[text: .newAxis], cache: cache, state: state)
        state = out.state
        return out.logits
    }

    /// The verify forward; also keeps the GDN layer inputs for `truncateCache`.
    func verifyForward(_ ids: [Int]) -> MLXArray {
        if ids.count > 1, state == nil, let qwen {
            let tokens = MLXArray(ids.map { Int32($0) })
            let (logits, saved) = qwen.verifyCompiledSaved(tokens[.newAxis], cache: cache)
            savedGDNInputs = saved
            return logits
        }
        savedGDNInputs = [:]
        return forward(ids)
    }

    /// After a partial accept, cut the verify forward's state back to the accepted prefix
    /// instead of restoring the snapshot and replaying the prefix. KV caches are trimmed;
    /// each GDN layer recomputes its conv/recurrent state over the prefix from the snapshot,
    /// using the layer inputs the verify forward saved.
    func truncateCache(to prefixRows: Int, snap: CacheSnapshot) {
        guard let qwen else { return }
        for e in snap.recurrent {
            guard let m = cache[e.index] as? MambaCache, let conv = e.conv, let rec = e.rec,
                let x = savedGDNInputs[e.index], prefixRows > 0
            else { continue }
            let (newConv, newRec) = qwen.gdnStateAfter(
                layerIndex: e.index, x: x[0..., ..<prefixRows, 0...], convState: conv,
                recState: rec)
            m[0] = newConv
            m[1] = newRec
            m.offset = e.offset + prefixRows
        }
        for e in snap.attention {
            let c = cache[e.index]
            let drop = c.offset - (e.offset + prefixRows)
            if drop > 0 { c.trim(drop) }
        }
    }

    /// Longest suffix match (up to `ngramMax` tokens) of `ctx` from `lo` on, continued for up
    /// to `limit` tokens.
    func lookupDraft(_ ctx: [Int], limit: Int, lo: Int) -> [Int] {
        let n = ctx.count
        guard n > 1, limit > 0 else { return [] }
        for m in stride(from: min(ngramMax, n - 1), through: 1, by: -1) {
            let suffix = ctx[(n - m)...]
            var p = n - m - 1
            while p >= lo {
                if ctx[p ..< (p + m)] == suffix {
                    let start = p + m
                    let end = min(n, start + limit)
                    return start < end ? Array(ctx[start ..< end]) : []
                }
                p -= 1
            }
        }
        return []
    }

    /// Penalized greedy pick of one row and the gap to the runner-up.
    func top2Pick(_ penalized: MLXArray) -> (token: Int, margin: Float) {
        let top2 = argPartition(-penalized, kth: 1, axis: -1)[0..., ..<2]
        let values = takeAlong(penalized, top2, axis: -1)
        eval([top2, values])
        let ids = top2.asType(.int32).asArray(Int32.self)
        let vals = values.asArray(Float.self)
        let firstWins = vals[0] >= vals[1]
        return (Int(ids[firstWins ? 0 : 1]), abs(vals[0] - vals[1]))
    }

    /// Whether the penalized pick of a near-tie row is also the raw argmax. Then a different
    /// rounding can at worst make the penalty decide differently; if the penalty flipped the
    /// raw order, the row is unsafe.
    func penaltyPreservedRawOrder(penalized: MLXArray, raw: MLXArray, pick: Int) -> Bool {
        let top2 = argPartition(-penalized, kth: 1, axis: -1)[0..., ..<2]
        let rawValues = takeAlong(raw, top2, axis: -1)
        eval([top2, rawValues])
        let ids = top2.asType(.int32).asArray(Int32.self)
        let rv = rawValues.asArray(Float.self)
        return Int(ids[rv[0] >= rv[1] ? 0 : 1]) == pick
    }

    /// Emit new tokens in order, stopping at a stop token or max_tokens. Returns true when the
    /// request is finished.
    func processNewTokens(_ tokens: [Int], _ tokenSteps: [StepLogprobs]) -> Bool {
        for (idx, token) in tokens.enumerated() {
            if token == unknownTokenID || stopTokenIDs.contains(token) {
                sampled.append(token)
                if topK > 0, idx < tokenSteps.count { steps.append(tokenSteps[idx]) }
                finishReason = "stop"
                stopToken = token
                return true
            }
            if generated.count >= payload.maxTokens {
                finishReason = "length"
                return true
            }
            generated.append(token)
            sampled.append(token)
            if topK > 0, idx < tokenSteps.count { steps.append(tokenSteps[idx]) }
            emitToken(token)
        }
        if generated.count >= payload.maxTokens {
            finishReason = "length"
            return true
        }
        return false
    }

    /// Finish the request with small forwards that are exact.
    ///
    /// A forward of up to 8 rows gives, row by row and in the state it leaves, exactly what 8
    /// single-token steps give: below 12 rows the quantized matmuls run one matvec per row, and
    /// up to 8 query rows the attention uses `sdpa_vector`, whose per-row sum depends only on
    /// the row's own keys. So each step here drafts a few tokens, checks them in one forward of
    /// at most 8 rows and keeps the accepted prefix, with the same result as plain decoding.
    /// The draft length follows the running acceptance rate: this path is taken when the model
    /// rewrites rather than copies, where long drafts are mostly wasted.
    func runSequentialTail() {
        let exactRows = 8
        var tailRate: Float = 0.3
        var draftCap = 1
        while true {
            while pending.count > exactRows {
                let out = forward([pending.removeFirst()])
                eval(out)
                eval(cache)
            }
            if generated.count >= payload.maxTokens {
                finishReason = "length"
                looping = false
                return
            }
            var limit = min(draftCap, payload.maxTokens - generated.count)
            if pending.count + limit > exactRows { limit = max(0, exactRows - pending.count) }
            let ctx = prompt + generated
            // Row-exactness holds while the attention stays below the 1024-key switch to the
            // two-pass vector kernel.
            let kvLength = cache.first(where: { $0 is KVCacheSimple })?.offset ?? 0
            var draft: [Int] = []
            if limit > 0, kvLength + pending.count + limit < 1024 {
                let lookup = lookupDraft(ctx, limit: limit, lo: userRegionStart)
                if lookup.count >= min(2, limit) {
                    draft = lookup
                } else if positionalPointer < prompt.count {
                    draft = Array(prompt[positionalPointer ..< min(prompt.count, positionalPointer + limit)])
                }
                if pending.count + draft.count > exactRows {
                    draft = Array(draft.prefix(max(0, exactRows - pending.count)))
                }
            }
            let input = pending + draft
            let base = pending.count - 1
            let snap = snapshotCache()
            let rows = forward(input)
            eval(rows)
            eval(cache)
            draftedTokens += draft.count
            rounds += 1
            var ring = params.processor()
            ring?.prompt(MLXArray(ctx.map { Int32($0) }))
            var accepted = 0
            var next: Int? = nil
            var newSteps: [StepLogprobs] = []
            for i in 0 ..< draft.count {
                let row = rows[0, base + i].reshaped([1, -1])
                let pick = sampler.sample(logits: ring?.process(logits: row) ?? row)
                let pickId = pick.item(Int.self)
                if let step = logprobStep(row: row, token: pick) { newSteps.append(step) }
                if pickId == draft[i] {
                    ring?.didSample(token: pick)
                    accepted += 1
                } else {
                    next = pickId
                    break
                }
            }
            if accepted == draft.count {
                let row = rows[0, base + draft.count].reshaped([1, -1])
                let pick = sampler.sample(logits: ring?.process(logits: row) ?? row)
                next = pick.item(Int.self)
                if let step = logprobStep(row: row, token: pick) { newSteps.append(step) }
            }
            acceptedDrafts += accepted
            var newTokens = Array(draft.prefix(accepted))
            if let next { newTokens.append(next) }
            if accepted == draft.count {
                pending = newTokens.count > accepted ? [newTokens[accepted]] : []
            } else {
                restoreCache(snap)
                pending += draft.prefix(accepted)
                if newTokens.count > accepted { pending.append(newTokens[accepted]) }
            }
            if processNewTokens(newTokens, newSteps) {
                looping = false
                return
            }
            advancePointer(over: newTokens)
            let rate = draft.isEmpty ? 0 : Float(accepted) / Float(draft.count)
            tailRate = 0.6 * tailRate + 0.4 * rate
            draftCap = tailRate < 0.15 ? 0 : tailRate < 0.4 ? 1 : tailRate < 0.7 ? 2 : 4
            if pending.isEmpty {
                looping = false
                return
            }
        }
    }

    // Warm-up only: trace the compiled verify forward for every row count the loop uses
    // (2...33). MLX compiles per shape on first use, which would otherwise cost the first
    // request that needs a shape ~6.6 ms. The cache is restored after each dummy forward.
    if payload.pretrace {
        let traceSnap = snapshotCache()
        for t in 2 ... 33 {
            let out = forward((0 ..< t).map { $0 % 97 })
            eval(out)
            eval(cache)
            restoreCache(traceSnap)
        }
    }

    let generateStart = Date()

    // The first token comes from the last row of the prefill, as in plain decoding.
    var firstProcessor = params.processor()
    firstProcessor?.prompt(MLXArray(prompt.map { Int32($0) }))
    let firstRow = logits[0..., -1, 0...]
    let firstToken = sampler.sample(logits: firstProcessor?.process(logits: firstRow) ?? firstRow)
        .item(Int.self)
    steps.append(contentsOf: logprobSteps(rows: firstRow, chosen: [firstToken]))
    sampled.append(firstToken)
    if firstToken == unknownTokenID || stopTokenIDs.contains(firstToken) {
        finishReason = "stop"
        stopToken = firstToken
    } else {
        generated.append(firstToken)
        emitToken(firstToken)
        advancePointer(over: [firstToken])
        if generated.count >= payload.maxTokens {
            finishReason = "length"
        }
    }
    pending = generated.isEmpty ? [] : [generated.last!]

    var draftCap = maxDraft
    var zeroRun = 0
    while looping && !pending.isEmpty && generated.count < payload.maxTokens {
        autoreleasepool {
            var limit = min(maxDraft, draftCap, payload.maxTokens - generated.count)
            // Near the end of the transcript, keep the round within 16 rows: a 16-row matmul
            // tile then covers it instead of a 32-row one.
            if prompt.count - positionalPointer <= 24 {
                limit = min(limit, 16 - pending.count)
            }
            if pending.count + limit > maxDraft {
                limit = max(0, maxDraft - pending.count)
            }
            let ctx = prompt + generated
            var draft: [Int] = []
            if limit > 0 {
                let lookup = lookupDraft(ctx, limit: limit, lo: userRegionStart)
                if lookup.count >= min(3, draftCap) {
                    draft = lookup
                } else if positionalPointer < prompt.count {
                    draft = Array(prompt[positionalPointer ..< min(prompt.count, positionalPointer + limit)])
                }
                if pending.count + draft.count > maxDraft {
                    draft = Array(draft.prefix(max(0, maxDraft - pending.count)))
                }
            }
            if draft.isEmpty {
                runSequentialTail()
                return
            }
            draftedTokens += draft.count
            let input = pending + draft
            let base = pending.count - 1
            let snap = snapshotCache()
            let rows = verifyForward(input)
            rounds += 1

            // Accept test. A row whose penalized top-2 margin is a near tie and whose pick is
            // not the raw argmax could flip under single-step rounding, so such a round rolls
            // back and finishes on the exact path.
            var accepted = 0
            var next: Int? = nil
            var uncertain = false
            if params.presencePenalty == nil, params.frequencyPenalty == nil,
                ctx.count >= params.repetitionContextSize
            {
                // All rows at once: penalty, top-2 and raw argmax are built on the lazy verify
                // logits and evaluated together with the cache update in one sync.
                let statsArrays = acceptStatsArrays(
                    rows: rows, base: base, ctx: ctx, draft: draft,
                    repetitionPenalty: params.repetitionPenalty,
                    contextSize: params.repetitionContextSize)
                eval(statsArrays.arrays + cache.flatMap { $0.state })
                let rowStats = statsArrays.decode()
                for i in 0 ..< draft.count {
                    let row = rowStats[i]
                    if row.margin < marginThreshold, row.rawArgmax != row.pick {
                        uncertain = true
                        break
                    }
                    if row.pick == draft[i] {
                        accepted += 1
                    } else {
                        next = row.pick
                        break
                    }
                }
                if !uncertain && accepted == draft.count {
                    let row = rowStats[draft.count]
                    if row.margin < marginThreshold && row.rawArgmax != row.pick {
                        uncertain = true
                    } else {
                        next = row.pick
                    }
                }
            } else {
                // Row by row with the repetition processor.
                eval(rows)
                eval(cache)
                var ring = params.processor()
                ring?.prompt(MLXArray(ctx.map { Int32($0) }))
                for i in 0 ... draft.count {
                    let row = rows[0, base + i].reshaped([1, -1])
                    let penalized = ring?.process(logits: row) ?? row
                    let (pick, margin) = top2Pick(penalized)
                    if margin < marginThreshold,
                        !penaltyPreservedRawOrder(penalized: penalized, raw: row, pick: pick)
                    {
                        uncertain = true
                        break
                    }
                    if i < draft.count, pick == draft[i] {
                        ring?.didSample(token: MLXArray([Int32(pick)]))
                        accepted += 1
                    } else {
                        next = pick
                        break
                    }
                }
            }

            if uncertain {
                restoreCache(snap)
                runSequentialTail()
                return
            }

            acceptedDrafts += accepted
            // Adapt the draft length to the acceptance rate. A verify forward of 9 to 16 rows
            // costs about the same (~38 ms, one 16-row matmul tile), so a poorly or partly
            // accepted round keeps up to 15 draft tokens; a well accepted one goes back to 32.
            let rate = Float(accepted) / Float(draft.count)
            zeroRun = accepted == 0 ? zeroRun + 1 : 0
            if rate < 0.25 {
                draftCap = max(2, min(draftCap, 15))
            } else if rate > 0.5 {
                draftCap = maxDraft
            } else {
                draftCap = 15
            }

            var newTokens = Array(draft.prefix(accepted))
            if let next { newTokens.append(next) }
            let newSteps =
                topK > 0
                ? logprobSteps(rows: rows[0, base ..< (base + newTokens.count)], chosen: newTokens)
                : []
            let emittedBefore = generated.count
            let finished = processNewTokens(newTokens, newSteps)
            advancePointer(over: Array(generated[emittedBefore...]))
            if finished {
                looping = false
                return
            }

            // Cache bookkeeping. All accepted: the forward already fed `input`, so only the
            // next token is pending. Partial: the accepted tokens are folded into the next
            // verify forward; once that would exceed `foldCap`, the state is cut back to the
            // accepted prefix instead (or, without saved GDN inputs, replayed).
            let nextPending = pending.count + accepted + (newTokens.count > accepted ? 1 : 0)
            if accepted == draft.count {
                pending = newTokens.count > accepted ? [newTokens[accepted]] : []
            } else if !savedGDNInputs.isEmpty, nextPending > foldCap {
                truncateCache(to: pending.count + accepted, snap: snap)
                pending = newTokens.count > accepted ? [newTokens[accepted]] : []
            } else {
                restoreCache(snap)
                pending += draft.prefix(accepted)
                if newTokens.count > accepted { pending.append(newTokens[accepted]) }
                if pending.count > foldCap {
                    eval(forward(Array(pending.dropLast())))
                    eval(cache)
                    rounds += 1
                    pending = [pending.last!]
                }
            }
            if zeroRun >= 2 {
                // Two rounds in a row accepted nothing: the model is rewriting, not copying.
                runSequentialTail()
                return
            }
        }
    }

    for output in toolProcessor.processEOSOutputs() {
        if case .response(let text) = output {
            emit(text)
        }
    }
    Stream().synchronize()
    let generateSeconds = Date().timeIntervalSince(generateStart)

    var logprobs: [[String: Any]]?
    if topK > 0 {
        let keep = min(steps.count, generated.count + (stopToken == nil ? 0 : 1))
        logprobs = (0 ..< keep).map { i in
            let ids = steps[i].indices.asType(.int32).asArray(Int32.self)
            let values = steps[i].values.asArray(Float.self)
            var top = zip(ids, values).map { (Int($0.0), Double($0.1)) }
            top.sort { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
            return [
                "token": sampled[i],
                "logprob": Double(steps[i].chosen.item(Float.self)),
                "top": top.map { [$0.0, $0.1] as [Any] },
            ]
        }
    }

    return GenerationOutcome(
        text: aggregated,
        tokens: generated,
        stopToken: stopToken,
        promptTokens: prompt.count,
        finishReason: finishReason,
        logprobs: logprobs,
        prefillSeconds: prefillSeconds,
        generateSeconds: generateSeconds,
        cachedPromptTokens: start,
        rounds: rounds,
        acceptedDrafts: acceptedDrafts,
        draftedTokens: draftedTokens)
}
