// f32 simdgroup_matrix GEMM for the encoder linears (same math as enc_gemm_splitk:
// C = A[M,K] (f32) x B[N,K]^T (bf16 weights) + bias, ascending-k accumulation,
// split-K partitions over k in [z*psize, (z+1)*psize)).
// Only the k-sum changes: simdgroup_float8x8 multiply-accumulate instead of 4x4 fma tiles.
#include <metal_stdlib>


using namespace metal;

template <int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void gemm_sg_body(device const float *A,
                             device const bfloat *B,
                             device float *C,
                             device const float *bias,
                             constant int &M,
                             constant int &N,
                             constant int &K,
                             constant int &psize,
                             constant int &fuse,
                             threadgroup float *smem,
                             uint3 tg,
                             uint3 tid,
                             uint warp) {
  constexpr int TM = BM / (8 * WM);  // 8x8 fragments per simdgroup along M
  constexpr int TN = BN / (8 * WN);
  constexpr int NTHR = WM * WN * 32;
  constexpr int PAD = 4;  // floats of row padding (16 B), as MLX steel
  constexpr int AS = BK + PAD, BS = BN + PAD;
  threadgroup float *As = smem;
  threadgroup float *Bs = smem + BM * AS;
  threadgroup float *Ct = smem;  // overlays As+Bs; safe after the last k barrier
  const int m0 = int(tg.y) * BM;
  const int n0 = int(tg.x) * BN;
  const int kz = int(tg.z);
  const int k0 = kz * psize;
  const int k1 = min(K, k0 + psize);
  const int t = int(tid.x);

  simdgroup_float8x8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  constexpr int AELEMS = BM * BK;
  constexpr int BELEMS = BK * BN;

  for (int kb = k0; kb < k1; kb += BK) {
    // A: BM*BK elements, 4 along k per thread.
#pragma unroll
    for (int e = t * 4; e < AELEMS; e += NTHR * 4) {
      const int row = e / BK, kx = e % BK;
      const int gm = m0 + row;
      float4 v = float4(0.0f);
      if (gm < M && kb + kx + 4 <= K)
        v = *(const device float4 *)(A + (size_t)gm * K + kb + kx);
      else if (gm < M)
        for (int i = 0; i < 4; ++i)
          v[i] = (kb + kx + i < K) ? A[(size_t)gm * K + kb + kx + i] : 0.0f;
      As[row * AS + kx + 0] = v.x;
      As[row * AS + kx + 1] = v.y;
      As[row * AS + kx + 2] = v.z;
      As[row * AS + kx + 3] = v.w;
    }
    // B: BK*BN elements; weights are [N,K] so read 4 consecutive k per output channel.
#pragma unroll
    for (int e = t * 4; e < BELEMS; e += NTHR * 4) {
      const int n = e / BK, kx = e % BK;
      const int gn = n0 + n;
      float4 v = float4(0.0f);
      if (gn < N && kb + kx + 4 <= K) {
        bfloat4 bv = *(const device bfloat4 *)(B + (size_t)gn * K + kb + kx);
        v = float4(float(bv.x), float(bv.y), float(bv.z), float(bv.w));
      } else if (gn < N) {
        for (int i = 0; i < 4; ++i)
          v[i] = (kb + kx + i < K) ? float(B[(size_t)gn * K + kb + kx + i]) : 0.0f;
      }
      Bs[(kx + 0) * BS + n] = v.x;
      Bs[(kx + 1) * BS + n] = v.y;
      Bs[(kx + 2) * BS + n] = v.z;
      Bs[(kx + 3) * BS + n] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int wm = (warp / WN) * (BM / WM);
    const int wn = (warp % WN) * (BN / WN);
#pragma unroll
    for (int k8 = 0; k8 < BK; k8 += 8) {
      simdgroup_float8x8 af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        simdgroup_load(af[i], As + (wm + i * 8) * AS + k8, AS);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        simdgroup_load(bf[j], Bs + k8 * BS + wn + j * 8, BS);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr int CS = BN + PAD;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      simdgroup_store(acc[i][j],
                      Ct + ((warp / WN) * (BM / WM) + i * 8) * CS + (warp % WN) * (BN / WN) + j * 8,
                      CS);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t zoff = fuse ? 0 : (size_t)kz * M * N;
  for (int e = t; e < BM * BN; e += NTHR) {
    const int r = e / BN, c = e % BN;
    const int m = m0 + r, n = n0 + c;
    if (m >= M || n >= N) continue;
    float v = Ct[r * CS + c];
    if (fuse) v += bias[n];
    C[zoff + (size_t)m * N + n] = v;
  }
}

#define GEMM_SG_KERNEL(NAME, BM, BN, BK, WM, WN)                                          \
  kernel void NAME(device const float *A [[buffer(0)]],                                   \
                   device const bfloat *B [[buffer(1)]],                                  \
                   device float *C [[buffer(2)]],                                         \
                   device const float *bias [[buffer(3)]],                                \
                   constant int &M [[buffer(4)]],                                         \
                   constant int &N [[buffer(5)]],                                         \
                   constant int &K [[buffer(6)]],                                         \
                   constant int &psize [[buffer(7)]],                                     \
                   constant int &fuse [[buffer(8)]],                                      \
                   uint3 tg [[threadgroup_position_in_grid]],                             \
                   uint3 tid [[thread_position_in_threadgroup]],                          \
                   uint warp [[simdgroup_index_in_threadgroup]]) {                        \
    constexpr int SMEM = (BM * (BK + 4) + BK * (BN + 4) > BM * (BN + 4))                  \
                             ? BM * (BK + 4) + BK * (BN + 4)                              \
                             : BM * (BN + 4);                                             \
    threadgroup float smem[SMEM];                                                         \
    gemm_sg_body<BM, BN, BK, WM, WN>(A, B, C, bias, M, N, K, psize, fuse, smem, tg, tid,  \
                                     warp);                                               \
  }

GEMM_SG_KERNEL(enc_gemm_sg32, 32, 32, 32, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg64, 32, 64, 32, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg6464, 64, 64, 32, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg6464k16, 64, 64, 16, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg6464w8, 64, 64, 32, 2, 4)
GEMM_SG_KERNEL(enc_gemm_sg32x128, 32, 128, 32, 2, 4)
GEMM_SG_KERNEL(enc_gemm_sg32k64, 32, 64, 64, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg16x64, 16, 64, 32, 2, 2)
GEMM_SG_KERNEL(enc_gemm_sg96x64, 96, 64, 32, 3, 2)
GEMM_SG_KERNEL(enc_gemm_sg32w14, 32, 64, 32, 1, 4)
GEMM_SG_KERNEL(enc_gemm_sg24x64, 24, 64, 32, 1, 4)
GEMM_SG_KERNEL(enc_gemm_sg40x64, 40, 64, 32, 1, 4)
GEMM_SG_KERNEL(enc_gemm_sg48x64, 48, 64, 32, 1, 4)
GEMM_SG_KERNEL(enc_gemm_sg24w8, 24, 64, 32, 1, 8)
GEMM_SG_KERNEL(enc_gemm_sg24x128, 24, 128, 32, 1, 8)
GEMM_SG_KERNEL(enc_gemm_sg24x32, 24, 32, 32, 1, 4)
GEMM_SG_KERNEL(enc_gemm_sg16x64w14, 16, 64, 32, 1, 4)
