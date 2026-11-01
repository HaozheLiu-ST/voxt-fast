// Decoder kernels for the Qwen3 text decoder, ported verbatim from MLX 0.32.3
// (kernels/rms_norm.metal, kernels/rope.metal, kernels/sdpa_vector.h) with unique
// kernel names so they can share one metallib with the other modules.
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// ---- RMSNorm (MLX rms_single_row, RMS_N_READS = 4) ----
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void dec_rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
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
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs using cached x values
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] =
          w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] =
            w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
  }
}

// ---- RoPE (MLX rope_single, forward = true, traditional = false) ----
template <typename T>
[[kernel]] void dec_rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset [[buffer(2)]],
    constant const float& scale [[buffer(3)]],
    constant const int64_t& stride [[buffer(4)]],
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  float L = scale * static_cast<float>(offset);
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  uint index_1 = pos.x + pos.y * uint(stride);
  uint index_2 = index_1 + grid.x;
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  out[index_1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  out[index_2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

// ---- SiLU: MLX compiled multiply(x, sigmoid(x)) in bf16 ----
template <typename T>
[[kernel]] void dec_silu(const device T* x [[buffer(0)]],
                         device T* out [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
  // MLX unary_ops.h Sigmoid: y = 1/(1+precise::exp(abs(x))); (x<0) ? y : 1-y,
  // all in T arithmetic, then the compiled graph multiplies in T.
  T v = x[gid];
  T y = T(1) / (T(1) + metal::precise::exp(metal::abs(v)));
  T s = (v < T(0)) ? y : T(1) - y;
  out[gid] = v * s;
}

// ---- elementwise add / multiply in bf16 ----
template <typename T>
[[kernel]] void dec_add(const device T* a [[buffer(0)]],
                        const device T* b [[buffer(1)]],
                        device T* out [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  out[gid] = static_cast<T>(static_cast<float>(a[gid]) + static_cast<float>(b[gid]));
}

template <typename T>
[[kernel]] void dec_mul(const device T* a [[buffer(0)]],
                        const device T* b [[buffer(1)]],
                        device T* out [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  out[gid] = static_cast<T>(static_cast<float>(a[gid]) * static_cast<float>(b[gid]));
}

// clang-format off
instantiate_kernel("dec_rms_bf16", dec_rms_single_row, bfloat16_t)
instantiate_kernel("dec_rope_single_bf16", dec_rope_single, bfloat16_t)
instantiate_kernel("dec_silu_bf16", dec_silu, bfloat16_t)
instantiate_kernel("dec_add_bf16", dec_add, bfloat16_t)
instantiate_kernel("dec_mul_bf16", dec_mul, bfloat16_t)
instantiate_kernel("dec_sdpa_vector_bf16_128_128", sdpa_vector, bfloat16_t, 128, 128)
// clang-format on

// KV cache row store: dst[(h*cap + pos)*hd + d] = src[h*hd + d]
template <typename T>
[[kernel]] void dec_kv_store(const device T* src [[buffer(0)]],
                             device T* dst [[buffer(1)]],
                             constant int& cap [[buffer(2)]],
                             constant int& pos [[buffer(3)]],
                             constant int& hd [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
  int h = gid / hd;
  int d = gid % hd;
  dst[(size_t(h) * cap + pos) * hd + d] = src[gid];
}
instantiate_kernel("dec_kv_store_bf16", dec_kv_store, bfloat16_t)
