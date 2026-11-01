// Minimal Metal runtime shared by all modules (owner: coordinator; extend via small, additive changes).
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu {

enum class DType : uint8_t { F32, F16, BF16, U32, I32, U8 };
size_t dtype_size(DType t);

// A view into an MTLBuffer. Shapes are row-major (last dim contiguous) unless a kernel says otherwise.
// Ownership is explicit (ARC does not manage object pointers stored in C++ aggregates): copies retain,
// the destructor releases, so a Tensor's buffer is freed as soon as its last view dies. A view created
// by hand (`t.buf = ...`) must take its own reference with CFRetain.
struct Tensor {
  __unsafe_unretained id<MTLBuffer> buf = nil;
  size_t offset = 0;  // bytes
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  Tensor() = default;
  Tensor(const Tensor &o) : buf(o.buf), offset(o.offset), dtype(o.dtype), shape(o.shape) {
    if (buf) CFRetain((__bridge CFTypeRef)buf);
  }
  Tensor &operator=(const Tensor &o) {
    if (this == &o) return *this;
    if (o.buf) CFRetain((__bridge CFTypeRef)o.buf);
    if (buf) CFRelease((__bridge CFTypeRef)buf);
    buf = o.buf;
    offset = o.offset;
    dtype = o.dtype;
    shape = o.shape;
    return *this;
  }
  Tensor(Tensor &&o) noexcept : buf(o.buf), offset(o.offset), dtype(o.dtype), shape(std::move(o.shape)) {
    o.buf = nil;
  }
  Tensor &operator=(Tensor &&o) noexcept {
    if (this == &o) return *this;
    if (buf) CFRelease((__bridge CFTypeRef)buf);
    buf = o.buf;
    offset = o.offset;
    dtype = o.dtype;
    shape = std::move(o.shape);
    o.buf = nil;
    return *this;
  }
  ~Tensor() {
    if (buf) CFRelease((__bridge CFTypeRef)buf);
  }
  int64_t numel() const;
  size_t nbytes() const { return (size_t)numel() * dtype_size(dtype); }
  int64_t dim(int i) const { return shape[i < 0 ? shape.size() + i : i]; }
  void *host() const { return (char *)[buf contents] + offset; }  // shared storage: CPU-visible
  template <class T> T *ptr() const { return (T *)host(); }
};

class Context {
 public:
  static Context &get();  // created on first use; loads asr.metallib from beside the executable
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  // Pipeline by kernel (function) name; cached. Throws std::runtime_error if missing.
  id<MTLComputePipelineState> pipeline(const std::string &name);
  // Pipeline specialised with function constants; `key` must uniquely identify the constant values.
  id<MTLComputePipelineState> pipeline(const std::string &name, const std::string &key,
                                       MTLFunctionConstantValues *constants);

  id<MTLBuffer> alloc(size_t bytes);  // MTLResourceStorageModeShared, zero-filled; +0 (ARC-managed)
  Tensor empty(std::vector<int64_t> shape, DType t);

 private:
  Context();
  NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipes_;
};

// Process-wide counters for the per-update cost bench (tests/bench_update.mm): how many command
// buffers / encoders / dispatches / waits an update costs. Incremented unconditionally (a few
// ns each); read+reset by the bench. Not used by the engine's own logic.
struct Stats {
  uint64_t cbs = 0;
  uint64_t encs = 0;
  uint64_t dispatches = 0;
  uint64_t waits = 0;
  double gpu_ms = 0;      // sum of every waited command buffer's GPUStart..GPUEnd
  double stage_ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // gpu_ms split by stats().stage
  int stage = 0;          // label set by the engine stages (0 = other)
};
Stats &stats();
void stats_reset();

// Labels the command buffers a stage waits for, so the bench can split GPU time per stage
// (0 = other, 1 = encoder, 2 = decoder prefill/forward_rows, 3 = decoder step).
struct StageScope {
  int prev;
  explicit StageScope(int s) : prev(stats().stage) { stats().stage = s; }
  ~StageScope() { stats().stage = prev; }
};

// One command buffer with one compute encoder, for recording a sequence of dispatches.
// Typical use:  gpu::Cmd c; c.enc ...; c.commit_and_wait();
// A Cmd that is destroyed without commit() ends its encoder (Metal asserts otherwise);
// reset() starts a fresh command buffer on the same object.
struct Cmd {
  __unsafe_unretained id<MTLCommandBuffer> cb = nil;
  __unsafe_unretained id<MTLComputeCommandEncoder> enc = nil;
  bool ended = false;
  Cmd();
  ~Cmd();
  Cmd(const Cmd &) = delete;
  Cmd &operator=(const Cmd &) = delete;
  void reset();            // end the current encoder (if open) and start a new command buffer
  void set(const Tensor &t, int index) { [enc setBuffer:t.buf offset:t.offset atIndex:index]; }
  template <class T> void set_value(const T &v, int index) { [enc setBytes:&v length:sizeof(T) atIndex:index]; }
  void set_bytes(const void *p, size_t n, int index) { [enc setBytes:p length:n atIndex:index]; }
  void use(const std::string &kernel) { [enc setComputePipelineState:Context::get().pipeline(kernel)]; }
  void use(id<MTLComputePipelineState> p) { [enc setComputePipelineState:p]; }
  void dispatch_groups(MTLSize groups, MTLSize threads_per_group) {
    ++stats().dispatches;
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads_per_group];
  }
  void dispatch_threads(MTLSize grid, MTLSize threads_per_group) {
    ++stats().dispatches;
    [enc dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
  }
  void barrier() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
  void commit();           // ends encoding and commits (non-blocking)
  void wait();             // waits for completion; throws on GPU error
  void commit_and_wait() { commit(); wait(); }
  double gpu_ms() const;   // GPU execution time of this command buffer (after wait)
};

// bf16 helpers for host-side reference code / tests.
inline float bf16_to_f32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f32_to_bf16(float f) {  // round to nearest even (as MLX / Metal bfloat)
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

double now_ms();  // monotonic wall clock, milliseconds

}  // namespace gpu
