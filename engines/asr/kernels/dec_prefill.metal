// Prefill (multi-row) kernels for the Qwen3 text decoder: MLX 0.32.3 rope (general,
// non-single), steel causal attention, KV row store and the split-K reduction,
// ported with unique kernel names.
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

using namespace metal;

// ---- RoPE (MLX rope_impl, forward = true, traditional = false, no head/seq transpose) ----
constant bool hs_transpose = false;
template <typename T, typename IdxT, int N, bool forward, bool traditional>
void dec_rope_impl(
    const device T* in,
    device T* out,
    const device int* offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    uint3 pos,
    uint3 grid) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);
  auto mat_idx = batch_idx * n_head + head_idx;

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  // Compute the input and output indices
  IdxT in_index_1;
  if (hs_transpose) {
    IdxT batch_stride = grid.y * IdxT(strides[1]);
    in_index_1 =
        batch_idx * batch_stride + pos.y * strides[1] + head_idx * strides[0];
  } else {
    in_index_1 = pos.y * IdxT(strides[1]) + mat_idx * IdxT(strides[0]);
  }
  IdxT in_index_2;
  IdxT out_index_1 =
      pos.y * IdxT(out_strides[1]) + mat_idx * IdxT(out_strides[0]);
  IdxT out_index_2;
  if (traditional) {
    out_index_1 += 2 * pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + 1;
    in_index_1 += 2 * pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + IdxT(strides[2]);
  } else {
    out_index_1 += pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + grid.x * IdxT(out_strides[2]);
    in_index_1 += pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + grid.x * IdxT(strides[2]);
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += IdxT(strides[0]);
    in_index_2 += IdxT(strides[0]);
    out_index_1 += IdxT(out_strides[0]);
    out_index_2 += IdxT(out_strides[0]);
  }
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void dec_rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  dec_rope_impl<T, IdxT, N, true, false>(
      in, out, offset, inv_freq, scale, strides, out_strides, offset_stride, n_head, pos, grid);
}

// ---- KV cache: store n rows starting at pos ----
template <typename T>
[[kernel]] void dec_kv_store_rows(const device T* src [[buffer(0)]],
                                  device T* dst [[buffer(1)]],
                                  constant int& cap [[buffer(2)]],
                                  constant int& pos [[buffer(3)]],
                                  constant int& hd [[buffer(4)]],
                                  constant int& nkv [[buffer(5)]],
                                  constant int& n [[buffer(6)]],
                                  constant int& head_major [[buffer(7)]],
                                  uint gid [[thread_position_in_grid]]) {
  int d = gid % hd;
  int h = (gid / hd) % nkv;
  int i = gid / (hd * nkv);
  // head_major: src is [nkv, n, hd] (rope output); else [n, nkv, hd].
  size_t s = head_major ? ((size_t)h * n + i) * hd + d : gid;
  dst[(size_t(h) * cap + pos + i) * hd + d] = src[s];
}

// ---- split-K sum: out[j] = sum_i float(p[i][j]) rounded to T (MLX strided sum) ----
template <typename T>
[[kernel]] void dec_splitk_sum(const device T* p [[buffer(0)]],
                               device T* out [[buffer(1)]],
                               constant int& split_k [[buffer(2)]],
                               constant int64_t& stride [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  // MLX col_reduce_small accumulates in the output type (bf16), sequentially.
  T acc = T(0);
  for (int i = 0; i < split_k; ++i) acc = p[(size_t)i * stride + gid] + acc;
  out[gid] = acc;
}

// clang-format off
instantiate_kernel("dec_rope_bf16", dec_rope, bfloat16_t, int32_t)
instantiate_kernel("dec_kv_store_rows_bf16", dec_kv_store_rows, bfloat16_t)
instantiate_kernel("dec_splitk_sum_bf16", dec_splitk_sum, bfloat16_t)
instantiate_kernel("dec_steel_attention_bf16_bq32_bk16_bd128_bv128_wm4_wn1",
                   attention, bfloat16_t, 32, 16, 128, 128, 4, 1, bfloat16_t, float)
// clang-format on
