// Per-row accept statistics for one speculative verify round.
//
// The verify forward produces logits for `pending + draft`; row `base + i` predicts the
// token after `ctx + draft[..<i]` (ctx = prompt + generated). The accept test needs, for
// every row, the penalized top-2 (pick + margin) and, for the near-tie policy, the raw
// argmax and whether each top-2 candidate is inside that row's repetition window.
//
// Everything is built lazily on top of the verify logits so the caller can put the verify
// forward, the penalty, the top-2 and the cache update into ONE eval / one stream sync
// (instead of eval(rows) + eval(cache) + one eval per row). The penalty math is exactly
// RepetitionContext.process: gather the row logits at the window indices, apply
// `x < 0 ? x * p : x / p`, scatter back. Duplicate window entries write identical values,
// so the order does not matter. The per-row penalized values are bit-identical to the
// per-row path; an exact top-2 tie has margin 0 and still falls back to sequential.

import Foundation
import MLX

/// Per-row accept statistics. `pick` is the penalized greedy token, `margin` the gap to the
/// second penalized candidate, `rawArgmax` the greedy token before the penalty.
nonisolated struct AcceptRowStats {
    var pick: Int
    var margin: Float
    var rawArgmax: Int
    var pickInWindow: Bool
    var secondInWindow: Bool
}

/// Lazy arrays for one verify round's accept stats; call `decode()` after the eval.
nonisolated struct AcceptStatsArrays {
    let top2: MLXArray  // [rowCount, 2] int32
    let top2Values: MLXArray  // [rowCount, 2] logit dtype
    let rawArgmax: MLXArray  // [rowCount] int32
    let inWindow: MLXArray  // [rowCount, 2] bool

    var arrays: [MLXArray] { [top2, top2Values, rawArgmax, inWindow] }

    func decode() -> [AcceptRowStats] {
        let ids = top2.asType(.int32).asArray(Int32.self)
        let vals = top2Values.asArray(Float.self)
        let raw = rawArgmax.asType(.int32).asArray(Int32.self)
        let win = inWindow.asArray(Bool.self)
        var out: [AcceptRowStats] = []
        out.reserveCapacity(raw.count)
        for i in 0 ..< raw.count {
            let a = Int(ids[2 * i]), b = Int(ids[2 * i + 1])
            let va = vals[2 * i], vb = vals[2 * i + 1]
            let firstWins = va >= vb
            out.append(
                AcceptRowStats(
                    pick: firstWins ? a : b,
                    margin: abs(va - vb),
                    rawArgmax: Int(raw[i]),
                    pickInWindow: win[2 * i + (firstWins ? 0 : 1)],
                    secondInWindow: win[2 * i + (firstWins ? 1 : 0)]))
        }
        return out
    }
}

/// Build the accept stats for `draft.count + 1` rows of the verify forward `rows`
/// ([1, T, V]). `base = pending.count - 1`; row i predicts after `ctx + draft[..<i]`.
nonisolated func acceptStatsArrays(
    rows: MLXArray, base: Int, ctx: [Int], draft: [Int],
    repetitionPenalty: Float?, contextSize: Int
) -> AcceptStatsArrays {
    let rowCount = draft.count + 1
    let rowSlice = rows[0, base ..< (base + rowCount)]  // [R, V]
    let history = ctx + draft
    let ctxCount = ctx.count
    let windowSize = max(1, contextSize)
    var window = [Int32]()
    window.reserveCapacity(rowCount * windowSize)
    for i in 0 ..< rowCount {
        let end = ctxCount + i
        let start = end - windowSize
        for j in start ..< end { window.append(Int32(history[j])) }
    }
    let idx = MLXArray(window).reshaped([rowCount, windowSize])
    let usePenalty = (repetitionPenalty ?? 0) != 0
    var penalized = rowSlice
    if usePenalty {
        let gathered = takeAlong(rowSlice, idx, axis: -1)
        let scaled = MLX.where(gathered .< 0, gathered * repetitionPenalty!, gathered / repetitionPenalty!)
        penalized = putAlong(rowSlice, idx, values: scaled, axis: -1)
    }
    // Top-2 without a full argPartition (which is ~5.9 ms for [33, 248320] here): the
    // pick is the same argMax the sampler uses (same tie-breaking), the second value is
    // the max with the pick's position masked out, and the pick's own value is a gather.
    let pick = argMax(penalized, axis: -1)  // [R]
    let pickIdx = pick[0..., .newAxis]  // [R, 1]
    let pickValue = max(penalized, axis: -1)  // [R]
    let vocabRange = MLXArray.arange(penalized.dim(-1))
    let secondMask = vocabRange[.newAxis, 0...] .== pickIdx  // [R, V]
    let negInf = MLXArray(-Float.infinity).asType(penalized.dtype)
    let masked = MLX.where(secondMask, negInf, penalized)
    let second = argMax(masked, axis: -1)  // [R] index of the second candidate
    let secondValue = max(masked, axis: -1)  // [R]
    let top2 = stacked([pick, second], axis: -1)  // [R, 2], pick first
    let top2Values = stacked([pickValue, secondValue], axis: -1)  // [R, 2]
    let rawArgmax = argMax(rowSlice, axis: -1)
    let inWindow: MLXArray
    if usePenalty {
        let match = idx[0..., 0..., .newAxis] .== top2[0..., .newAxis, 0...]  // [R, C, 2]
        inWindow = any(match, axis: -2)  // [R, 2]
    } else {
        inWindow = MLXArray.zeros([rowCount, 2], type: Bool.self)
    }
    return AcceptStatsArrays(
        top2: top2, top2Values: top2Values, rawArgmax: rawArgmax, inWindow: inWindow)
}
