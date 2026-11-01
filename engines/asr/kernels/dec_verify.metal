// Streaming-verification kernels: per-row top1/top2 over bf16 logits (argmax +
// top1-top2 margin) and an f32 -> bf16 embedding splice (MLX astype semantics).
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Merge candidate B (v2, i2) into A (v, i, v2) with first-max-wins on ties.
#define DEC_MERGE(v, v2, i, ov, ov2, oi)          \
  do {                                            \
    float _ov = (ov), _ov2 = (ov2);               \
    uint _oi = (oi);                              \
    if (_ov > (v) || (_ov == (v) && _oi < (i))) { \
      (v2) = max((v), _ov2);                      \
      (v) = _ov;                                  \
      (i) = _oi;                                  \
    } else {                                      \
      (v2) = max((v2), _ov);                      \
    }                                             \
  } while (0)

// Per-row top-1/top-2 over bf16 logits. Grid: one threadgroup of 1024 per row.
// out_idx[row] = first index attaining the maximum (MLX argmax rule);
// out_margin[row] = top1 - top2 as float (0 when the maximum is tied).
template <typename T>
[[kernel]] void dec_top2(const device T* x [[buffer(0)]], device uint* out_idx [[buffer(1)]],
                         device float* out_margin [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]],
                         uint simd_lane_id [[thread_index_in_simdgroup]],
                         uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float s1[32];
  threadgroup float s2[32];
  threadgroup uint si[32];
  const device T* r = x + (size_t)row * n;
  float best = -INFINITY, second = -INFINITY;
  uint bidx = 0xffffffffu;
  for (uint i = lid; i < n; i += tg_size) {
    float v = static_cast<float>(r[i]);
    if (v > best || (v == best && i < bidx)) {
      second = best;
      best = v;
      bidx = i;
    } else if (v > second) {
      second = v;
    }
  }
  // reduce within the simdgroup
  for (int off = 16; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(best, off);
    float ov2 = simd_shuffle_down(second, off);
    uint oi = simd_shuffle_down(bidx, off);
    DEC_MERGE(best, second, bidx, ov, ov2, oi);
  }
  if (simd_lane_id == 0) {
    s1[simd_group_id] = best;
    s2[simd_group_id] = second;
    si[simd_group_id] = bidx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    uint ng = tg_size / 32;
    float v = simd_lane_id < ng ? s1[simd_lane_id] : -INFINITY;
    float v2 = simd_lane_id < ng ? s2[simd_lane_id] : -INFINITY;
    uint i = simd_lane_id < ng ? si[simd_lane_id] : 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(v, off);
      float ov2 = simd_shuffle_down(v2, off);
      uint oi = simd_shuffle_down(i, off);
      DEC_MERGE(v, v2, i, ov, ov2, oi);
    }
    if (simd_lane_id == 0) {
      out_idx[row] = i;
      out_margin[row] = v - v2;
    }
  }
}

// Encoder rows f32 [rows, cols] -> bf16 at dst row `dst_row0` (MLX astype cast).
template <typename T>
[[kernel]] void dec_splice_rows(const device float* src [[buffer(0)]], device T* dst [[buffer(1)]],
                                constant uint& cols [[buffer(2)]],
                                constant uint& dst_row0 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  dst[(size_t)dst_row0 * cols + gid] = static_cast<T>(src[gid]);
}

// clang-format off
instantiate_kernel("dec_top2_bf16", dec_top2, bfloat16_t)
instantiate_kernel("dec_splice_f32_bf16", dec_splice_rows, bfloat16_t)
// clang-format on

// Same as dec_gather_rows (dec_gemv.metal) but the ids live in the constant address
// space (Cmd::set_bytes copies them into the command buffer at encode time), so
// several embed calls can be recorded into ONE command buffer without clobbering
// a shared host-written ids buffer.
template <typename T>
[[kernel]] void dec_gather_rows_c(const device uint32_t* w [[buffer(0)]],
                                  const device T* scales [[buffer(1)]],
                                  const device T* biases [[buffer(2)]],
                                  constant int32_t* ids [[buffer(3)]],
                                  device uint32_t* w_out [[buffer(4)]], device T* s_out [[buffer(5)]],
                                  device T* b_out [[buffer(6)]], constant int& n [[buffer(7)]],
                                  constant int& packed_per_row [[buffer(8)]],
                                  constant int& groups_per_row [[buffer(9)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.y) >= n) return;
  const int row = ids[gid.y];
  const int i = gid.x;
  if (i < packed_per_row) {
    w_out[(size_t)gid.y * packed_per_row + i] = w[(size_t)row * packed_per_row + i];
  } else {
    const int g = i - packed_per_row;
    s_out[(size_t)gid.y * groups_per_row + g] = scales[(size_t)row * groups_per_row + g];
    b_out[(size_t)gid.y * groups_per_row + g] = biases[(size_t)row * groups_per_row + g];
  }
}
instantiate_kernel("dec_gather_rows_c_bf16", dec_gather_rows_c, bfloat16_t)
