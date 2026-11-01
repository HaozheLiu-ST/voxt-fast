// Small shared utility kernels (owner: coordinator).
#include <metal_stdlib>
using namespace metal;

kernel void copy_u32(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                     constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}

kernel void bf16_to_f32(device const bfloat *src [[buffer(0)]], device float *dst [[buffer(1)]],
                        constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = float(src[i]);
}
