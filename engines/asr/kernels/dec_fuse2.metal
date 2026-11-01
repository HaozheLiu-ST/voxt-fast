// More fused decode kernels: q/k rmsnorm in one dispatch, q/k rope with the k
// result written straight into the KV cache, and a device argmax (first max wins).
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

// One row of MLX rms_single_row (RMS_N_READS = 4, float acc, precise::rsqrt).
template <typename T, int N_READS = RMS_N_READS>
METAL_FUNC void dec_rms_row_body(const device T* x, const device T* w, device T* out,
                                 float eps, uint axis_size, uint row,
                                 threadgroup float* local_inv_mean,
                                 threadgroup float* local_sums, uint lid,
                                 uint simd_lane_id, uint simd_group_id) {
  constexpr int SIMD_SIZE = 32;
  float acc = 0;
  float thread_x[N_READS];
  x += row * size_t(axis_size) + lid * N_READS;
  w += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out += row * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++)
      out[i] = w[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
  } else {
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < axis_size)
        out[i] = w[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
  }
}

// q rows [0,nq) then k rows [0,nk) in one dispatch (one threadgroup per row).
template <typename T>
[[kernel]] void dec_rms_qk(const device T* xq [[buffer(0)]], const device T* wq [[buffer(1)]],
                           device T* outq [[buffer(2)]], const device T* xk [[buffer(3)]],
                           const device T* wk [[buffer(4)]], device T* outk [[buffer(5)]],
                           constant float& eps [[buffer(6)]],
                           constant uint& axis_size [[buffer(7)]], constant uint& nq [[buffer(8)]],
                           uint gid [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint simd_lane_id [[thread_index_in_simdgroup]],
                           uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  if (gid < nq) {
    dec_rms_row_body<T>(xq, wq, outq, eps, axis_size, gid, local_inv_mean, local_sums, lid,
                        simd_lane_id, simd_group_id);
  } else {
    dec_rms_row_body<T>(xk, wk, outk, eps, axis_size, gid - nq, local_inv_mean, local_sums, lid,
                        simd_lane_id, simd_group_id);
  }
}

// rope_single for q heads [0,nq) and k heads [nq,nq+nk); the k result goes
// directly to the KV cache row at `offset` (same math as dec_rope_single).
template <typename T>
[[kernel]] void dec_rope_qk_store(
    const device T* inq [[buffer(0)]], device T* outq [[buffer(1)]],
    const device T* ink [[buffer(2)]], device T* kcache [[buffer(3)]],
    constant const int& offset [[buffer(4)]], constant const float& scale [[buffer(5)]],
    constant const int64_t& stride [[buffer(6)]], constant const int& nq [[buffer(7)]],
    constant const int& cap [[buffer(8)]], constant const int& hd [[buffer(9)]],
    constant const float& base [[buffer(10)]], uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int h = (int)pos.y;
  const int hl = h < nq ? h : h - nq;
  const device T* src = h < nq ? inq : ink;
  const uint index_1 = pos.x + uint(hl) * uint(stride);
  const uint index_2 = index_1 + grid.x;
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  float L = scale * static_cast<float>(offset);
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  float x1 = static_cast<float>(src[index_1]);
  float x2 = static_cast<float>(src[index_2]);
  T o1 = static_cast<T>(x1 * costheta - x2 * sintheta);
  T o2 = static_cast<T>(x1 * sintheta + x2 * costheta);
  if (h < nq) {
    outq[index_1] = o1;
    outq[index_2] = o2;
  } else {
    device T* dst = kcache + (size_t(hl) * cap + uint(offset)) * uint(hd);
    dst[pos.x] = o1;
    dst[pos.x + grid.x] = o2;
  }
}

// ---- argmax over bf16 logits (MLX argmax: first maximum wins) ----
// One threadgroup; each thread strides over the array, then a tree reduction
// that prefers the lower index on ties.
template <typename T>
[[kernel]] void dec_argmax(const device T* x [[buffer(0)]], device uint* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint tg_size [[threads_per_threadgroup]],
                           uint simd_lane_id [[thread_index_in_simdgroup]],
                           uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sval[32];
  threadgroup uint sidx[32];
  float best = -INFINITY;
  uint bidx = 0xffffffffu;
  for (uint i = lid; i < n; i += tg_size) {
    float v = static_cast<float>(x[i]);
    if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
  }
  // reduce within the simdgroup
  for (int off = 16; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(best, off);
    uint oi = simd_shuffle_down(bidx, off);
    if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
  }
  if (simd_lane_id == 0) { sval[simd_group_id] = best; sidx[simd_group_id] = bidx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    uint ng = tg_size / 32;
    float v = simd_lane_id < ng ? sval[simd_lane_id] : -INFINITY;
    uint i = simd_lane_id < ng ? sidx[simd_lane_id] : 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(v, off);
      uint oi = simd_shuffle_down(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (simd_lane_id == 0) out[0] = i;
  }
}

// clang-format off
instantiate_kernel("dec_rms_qk_bf16", dec_rms_qk, bfloat16_t)
instantiate_kernel("dec_rope_qk_store_bf16", dec_rope_qk_store, bfloat16_t)
instantiate_kernel("dec_argmax_bf16", dec_argmax, bfloat16_t)
// clang-format on

// [nh, n, hd] (head-major, sdpa_vector output) -> [n, nh, hd] (seq-major, o_proj input)
template <typename T>
[[kernel]] void dec_transpose_nh(const device T* in [[buffer(0)]], device T* out [[buffer(1)]],
                                 constant int& n [[buffer(2)]], constant int& nh [[buffer(3)]],
                                 constant int& hd [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
  int d = gid % hd;
  int t = (gid / hd) % n;
  int h = gid / (hd * n);
  out[gid] = in[((size_t)h * n + t) * hd + d];
}
instantiate_kernel("dec_transpose_nh_bf16", dec_transpose_nh, bfloat16_t)
