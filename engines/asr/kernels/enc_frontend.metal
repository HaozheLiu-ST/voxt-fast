// Audio front end kernels (owner: encoder worker).
// Reflect pad + window + power/mel/log10 + dynamic range, matching qwen3_audio.cpp LogMel
// (voxt_swift layout) and common/audio.cpp.
#include <metal_stdlib>
using namespace metal;

// Padded sample q of the reference's `padded` array (length n + 400):
//   q < 200            -> audio[200 - q]        (left reflect, skips sample 0)
//   200 <= q < 200+n   -> audio[q - 200]
//   q >= 200+n         -> audio[2n + 198 - q]   (right reflect, skips the last sample)
kernel void enc_pad(device const float *audio [[buffer(0)]],
                    device float *pad [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    constant int &p0 [[buffer(3)]],
                    constant uint &len [[buffer(4)]],
                    uint i [[thread_position_in_grid]]) {
  if (i >= len) return;
  const int q = p0 + int(i);
  int idx;
  if (q < 200) {
    idx = 200 - q;
  } else if (q < int(n) + 200) {
    idx = q - 200;
  } else {
    idx = 2 * int(n) + 198 - q;
  }
  pad[i] = (idx >= 0 && idx < int(n)) ? audio[idx] : 0.0f;
}

// frames[f][j] = pad[f * 160 + j] * hann[j]
kernel void enc_window(device const float *pad [[buffer(0)]],
                       device const float *hann [[buffer(1)]],
                       device float *frames [[buffer(2)]],
                       constant uint &frame_count [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.y;
  const uint j = gid.x;
  if (f >= frame_count || j >= 400) return;
  frames[f * 400 + j] = pad[f * 160 + j] * hann[j];
}

// One mel bin of one frame: sum_k |X[k]|^2 * filter[k][m], log10(max(v, 1e-10)).
// Also reduces the per-frame max into frame_max[f].
kernel void enc_mel_log(device const float2 *spec [[buffer(0)]],
                        device const float *filters [[buffer(1)]],
                        device float *raw [[buffer(2)]],
                        device float *frame_max [[buffer(3)]],
                        constant uint &frame_count [[buffer(4)]],
                        uint m [[thread_position_in_threadgroup]],
                        uint f [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  if (f >= frame_count || m >= 128) return;
  float acc = 0.0f;
  for (int k = 0; k < 201; ++k) {
    const float2 x = spec[f * 201 + k];
    const float r = metal::precise::sqrt(x.x * x.x + x.y * x.y);
    acc += (r * r) * filters[k * 128 + m];
  }
  const float v = metal::precise::log10(metal::max(acc, 1e-10f));
  raw[f * 128 + m] = v;
  // max over the 128 bins of this frame (4 simdgroups of 32)
  float mx = simd_max(v);
  threadgroup float partial[4];
  if (lane == 0) partial[m / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (m == 0) {
    frame_max[f] = metal::max(metal::max(partial[0], partial[1]),
                              metal::max(partial[2], partial[3]));
  }
}

// Single-threadgroup max over frame_max[0..n).
kernel void enc_max_reduce(device const float *frame_max [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant uint &n [[buffer(2)]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint lsize [[threads_per_threadgroup]]) {
  threadgroup float partial[32];
  float mx = -INFINITY;
  for (uint i = tid; i < n; i += lsize) mx = metal::max(mx, frame_max[i]);
  mx = simd_max(mx);
  if (lane == 0) partial[tid / 32] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float r = -INFINITY;
    const uint groups = (lsize + 31) / 32;
    for (uint i = 0; i < groups; ++i) r = metal::max(r, partial[i]);
    out[0] = r;
  }
}

// mel[m][f] = (max(raw[f][m], clamp_max - 8) + 4) / 4
kernel void enc_mel_finalize(device const float *raw [[buffer(0)]],
                             device const float *clamp_max [[buffer(1)]],
                             device float *mel [[buffer(2)]],
                             constant uint &frame_count [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint m = gid.y;
  if (f >= frame_count || m >= 128) return;
  const float x = metal::max(raw[f * 128 + m], clamp_max[0] - 8.0f);
  mel[m * frame_count + f] = (x + 4.0f) / 4.0f;
}
