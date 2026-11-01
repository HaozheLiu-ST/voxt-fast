// Qwen3 text decoder (6-bit affine) for the ASR engine.
// Numerics follow MLX 0.32.3 exactly : bf16 activations,
// quantized_matmul(transpose=true), fast::rms_norm, fast::rope (non-traditional),
// fast::scaled_dot_product_attention (sdpa_vector for one row), tied LM head.
#pragma once

#include <string>
#include <vector>

#include "gpu.h"
#include "weights.h"

namespace dec {

struct Config {
  int n_layers = 28;
  int n_heads = 16;
  int n_kv = 8;
  int head_dim = 128;
  int hidden = 2048;
  int inter = 6144;
  int vocab = 151936;
  float eps = 1e-6f;
  float rope_theta = 1e6f;
};

// Per-layer KV cache, [n_kv, cap, head_dim] bf16, explicit length that can be
// truncated (rollback to position p) and extended.
struct KVCache {
  gpu::Tensor keys, values;
  int len = 0;
  int cap = 0;
  void reserve(int new_cap, int n_kv, int head_dim);
  void truncate(int p) { len = p; }
};

class Decoder {
 public:
  explicit Decoder(const Weights &w);
  const Config &config() const { return cfg_; }

  // Embedding rows for `ids` -> bf16 [n, hidden] (dequantized tied embedding table).
  gpu::Tensor embed(const std::vector<int> &ids);
  // Positions cache.len .. cache.len+n-1; returns last-layer hidden states [n, hidden].
  // n == 1 is the decode path (bit-exact vs the reference dumps).
  gpu::Tensor forward(const gpu::Tensor &embeddings, int n, std::vector<KVCache> &caches);
  // Tied LM head for one row of hidden states -> logits [vocab] bf16.
  gpu::Tensor logits(const gpu::Tensor &hidden_row);
  // One full decode step in a single command buffer: embed(id) + 28 layers +
  // final norm + LM head + device argmax. Returns the next token. *logits_out
  // (optional) receives the logits view, valid until the next step.
  int step(int id, std::vector<KVCache> &caches, gpu::Tensor *logits_out = nullptr);
  // First-max-wins argmax over bf16 logits (MLX tie rule).
  int argmax(const gpu::Tensor &logits);

  // ---- Streaming-verification API ----
  // Per-row greedy result of a batched forward: argmax (MLX first-max-wins) and
  // confidence = top1 - top2 of the bf16 logits, as float.
  struct RowOut {
    int argmax = 0;
    float margin = 0.0f;
  };
  // Batched forward over n rows at KV positions offset..offset+n-1 (every cache is
  // rolled back to `offset` first), then final norm + tied LM head over the LAST k
  // rows (k <= 0 -> n) with a device top1/top2 kernel, all in ONE command buffer.
  // Buffers are cached (no per-call allocations once k has been seen). *logits_out
  // (optional) receives the [k, vocab] bf16 logits, valid until the next call.
  std::vector<RowOut> forward_rows(const gpu::Tensor &embeddings, int n,
                                   std::vector<KVCache> &caches, int offset, int k = 0,
                                   gpu::Tensor *logits_out = nullptr);
  // Decode step like step() but also reports the top1-top2 margin (float).
  int step_margin(int id, std::vector<KVCache> &caches, float *margin,
                  gpu::Tensor *logits_out = nullptr);
  // Preallocate KV capacity for every layer (growth preserves the first len rows).
  void reserve_kv(std::vector<KVCache> &caches, int cap);
  // Cheap rollback of every layer to prefix length p (no GPU work).
  void rollback(std::vector<KVCache> &caches, int p);
  // Embedding buffer [rows, hidden] bf16 for prompt + audio + draft text.
  gpu::Tensor embedding_buffer(int rows);
  // Encoder rows (f32 [rows, hidden]) -> bf16 into emb at row_offset, as MLX astype.
  void splice_embeddings(gpu::Cmd &c, gpu::Tensor &emb, const gpu::Tensor &f32_rows,
                         int row_offset, int rows);
  void splice_embeddings(gpu::Tensor &emb, const gpu::Tensor &f32_rows, int row_offset,
                         int rows);
  // Embed ids into emb rows [row_offset, row_offset+n) (gather + dequantize).
  void embed_rows_into(gpu::Cmd &c, const std::vector<int> &ids, gpu::Tensor &emb,
                       int row_offset);

  // Timing: GPU time of the last forward()/logits() command buffer (ms).
  double last_gpu_ms = 0;
  // Use the fused decode path (default). false = the unfused reference-order path
  // used by tests to dump every intermediate.
  bool use_fused = true;
  // qmv threadgroup shape for the fused GEMVs: 0 = 2x4 (MLX), 1 = 4x2.
  int qmv_shape = 0;

  // Debug hooks used by tests: dump intermediate tensors of the last forward call.
  struct Trace {
    std::vector<gpu::Tensor> tensors;
    std::vector<std::string> names;
    bool enabled = false;
    void add(const std::string &name, const gpu::Tensor &t);
  } trace;

 private:
  gpu::Tensor forward_decode(const gpu::Tensor &embeddings, std::vector<KVCache> &caches);
  gpu::Tensor forward_prefill(const gpu::Tensor &embeddings, int n, std::vector<KVCache> &caches);
  // Records the 28-layer prefill into `c` (no commit); returns the last hidden [n, hidden].
  gpu::Tensor prefill_layers(gpu::Cmd &c, const gpu::Tensor &embeddings, int n,
                             std::vector<KVCache> &caches);
  // Cached [k, vocab] logits / [k, hidden] normed / uint[k] idx / float[k] margin.
  void ensure_rows_buffers(int k, int vocab);
  // Records x[M,K] @ W[N,K]^T (6-bit) into c (qmm_t / qmm_t_splitk as MLX dispatches).
  void qmm_into(gpu::Cmd &c, const gpu::Tensor &x, const std::string &prefix, int M, int N, int K,
                gpu::Tensor &y);
  gpu::Tensor rms_norm(const gpu::Tensor &x, const std::string &w, int rows, int axis);
  // Bump allocation from the per-step arena (falls back to a fresh buffer while
  // tracing, so dumped tensors stay valid).
  gpu::Tensor buf(std::vector<int64_t> shape);
  void arena_reset();  // rewind to the per-step keep mark (per-layer reuse)
  gpu::Tensor decode_layers_fused(gpu::Cmd &c, const gpu::Tensor &embeddings,
                                  std::vector<KVCache> &caches);
  gpu::Tensor decode_layers_traced(const gpu::Tensor &embeddings, std::vector<KVCache> &caches);
  void embed_into(gpu::Cmd &c, const std::vector<int> &ids, gpu::Tensor &out);
  gpu::Tensor qmv(const gpu::Tensor &x, const std::string &prefix, int n, int k);
  void rope(gpu::Tensor &t, int n_heads, int offset);

  const Weights &w_;
  Config cfg_;
  gpu::Tensor ids_buf_, gather_w_, gather_s_, gather_b_;
  int gather_cap_ = 0;
  gpu::Tensor arena_, argmax_buf_, emb_buf_, logits_buf_, normed_buf_;
  int64_t arena_off_ = 0, arena_keep_ = 0;
  gpu::Tensor splitk_temp_;  // cached split-K GEMM partials (qmm_into)
  gpu::Tensor rows_logits_, rows_normed_, rows_idx_, rows_margin_, margin_buf_;
  int rows_cap_ = 0, rows_vocab_ = 0;
};

}  // namespace dec
