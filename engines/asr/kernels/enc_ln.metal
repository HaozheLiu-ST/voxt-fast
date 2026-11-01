// LayerNorm for the audio encoder, ported verbatim from MLX 0.32.3
// mlx/backend/metal/kernels/layer_norm.metal (layer_norm_single_row), so the reduction
// order matches what the reference (mx::fast::layer_norm, axis_size 1024, n_reads 8,
// 128 threads) does.  Weights/biases are bf16 in the checkpoint and are converted to
// float exactly, as MLX's dtype promotion does.
#include <metal_stdlib>
using namespace metal;

template <int N = 1>
inline void enc_ln_initialize_buffer(threadgroup float *xs,
                                     uint simd_lane_id [[thread_index_in_simdgroup]],
                                     uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (simd_group_id == 0) {
    for (int i = 0; i < N; i++) xs[N * simd_lane_id + i] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int N = 1>
inline void enc_ln_threadgroup_sum(thread float *x, threadgroup float *xs,
                                   uint simd_lane_id [[thread_index_in_simdgroup]],
                                   uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  for (int i = 0; i < N; i++) x[i] = simd_sum(x[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    for (int i = 0; i < N; i++) xs[N * simd_group_id + i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N; i++) {
    x[i] = xs[N * simd_lane_id + i];
    x[i] = simd_sum(x[i]);
  }
}

// One threadgroup of 128 threads per row, 8 contiguous elements per thread (axis 1024).
kernel void enc_layer_norm(const device float *x [[buffer(0)]],
                           const device bfloat *w [[buffer(1)]],
                           const device bfloat *b [[buffer(2)]],
                           device float *out [[buffer(3)]],
                           constant float &eps [[buffer(4)]],
                           constant uint &axis_size [[buffer(5)]],
                           uint gid [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint simd_lane_id [[thread_index_in_simdgroup]],
                           uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  constexpr int N_READS = 8;

  float thread_x[N_READS] = {0};
  threadgroup float local_buffer[SIMD_SIZE] = {0};
  enc_ln_initialize_buffer(local_buffer, simd_lane_id, simd_group_id);

  x += gid * size_t(axis_size) + lid * N_READS;
  out += gid * size_t(axis_size) + lid * N_READS;

  const bool safe = lid * N_READS + N_READS <= axis_size;
  const int n = axis_size - lid * N_READS;

  if (safe) {
    for (int i = 0; i < N_READS; i++) thread_x[i] = x[i];
  } else {
    for (int i = 0; i < n; i++) thread_x[i] = x[i];
  }

  float mean = 0;
  for (int i = 0; i < N_READS; i++) mean += thread_x[i];
  enc_ln_threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  float normalizer = 0;
  if (!safe) {
    for (int i = n; i < N_READS; i++) thread_x[i] = mean;
  }
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    normalizer += thread_x[i] * thread_x[i];
  }
  enc_ln_threadgroup_sum(&normalizer, local_buffer, simd_lane_id, simd_group_id);
  normalizer = metal::precise::rsqrt(normalizer / axis_size + eps);

  const uint base = lid * N_READS;
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      out[i] = float(w[base + i]) * thread_x[i] + float(b[base + i]);
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      out[i] = float(w[base + i]) * thread_x[i] + float(b[base + i]);
    }
  }
}

// out += bias[n] for a [M, N] f32 matrix (residual add of a Linear result).
kernel void enc_add_bias(device float *a [[buffer(0)]], device const float *bias [[buffer(1)]],
                         constant uint &N [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x, m = gid.y;
  if (n >= N) return;
  a[(size_t)m * N + n] += bias[n];
}
