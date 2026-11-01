#include "gpu.h"

#include <mach-o/dyld.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace gpu {

size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: case DType::U32: case DType::I32: return 4;
    case DType::F16: case DType::BF16: return 2;
    case DType::U8: return 1;
  }
  return 0;
}

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

static std::string exe_dir() {
  char buf[4096];
  uint32_t n = sizeof(buf);
  if (_NSGetExecutablePath(buf, &n) != 0) return ".";
  char real[PATH_MAX];
  if (!realpath(buf, real)) return ".";
  std::string p(real);
  auto s = p.rfind('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

Context &Context::get() {
  static Context *c = new Context();
  return *c;
}

Context::Context() {
  device = MTLCreateSystemDefaultDevice();
  if (!device) throw std::runtime_error("no Metal device");
  queue = [device newCommandQueue];
  std::string path = exe_dir() + "/asr.metallib";
  NSError *err = nil;
  library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]]
                                error:&err];
  if (!library) throw std::runtime_error("cannot load " + path + ": " + (err ? err.localizedDescription.UTF8String : ""));
  pipes_ = [NSMutableDictionary dictionary];
}

id<MTLComputePipelineState> Context::pipeline(const std::string &name) {
  NSString *k = [NSString stringWithUTF8String:name.c_str()];
  id<MTLComputePipelineState> p = pipes_[k];
  if (p) return p;
  id<MTLFunction> f = [library newFunctionWithName:k];
  if (!f) throw std::runtime_error("kernel not found in metallib: " + name);
  NSError *err = nil;
  p = [device newComputePipelineStateWithFunction:f error:&err];
  if (!p) throw std::runtime_error("pipeline " + name + ": " + (err ? err.localizedDescription.UTF8String : ""));
  pipes_[k] = p;
  return p;
}

id<MTLComputePipelineState> Context::pipeline(const std::string &name, const std::string &key,
                                              MTLFunctionConstantValues *constants) {
  NSString *k = [NSString stringWithUTF8String:(name + "#" + key).c_str()];
  id<MTLComputePipelineState> p = pipes_[k];
  if (p) return p;
  NSError *err = nil;
  id<MTLFunction> f = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                                    constantValues:constants
                                             error:&err];
  if (!f) throw std::runtime_error("kernel " + name + " (" + key + "): " + (err ? err.localizedDescription.UTF8String : ""));
  p = [device newComputePipelineStateWithFunction:f error:&err];
  if (!p) throw std::runtime_error("pipeline " + name + ": " + (err ? err.localizedDescription.UTF8String : ""));
  pipes_[k] = p;
  return p;
}

id<MTLBuffer> Context::alloc(size_t bytes) {
  size_t n = bytes ? bytes : 16;
  id<MTLBuffer> b = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
  if (!b) throw std::runtime_error("buffer allocation failed");
  memset([b contents], 0, n);
  return b;
}

Tensor Context::empty(std::vector<int64_t> shape, DType t) {
  Tensor x;
  x.shape = std::move(shape);
  x.dtype = t;
  // Allocate here (not through a function returning id, which would autorelease the buffer): the
  // Tensor owns the only reference, so the memory is freed as soon as the last view dies.
  size_t bytes = x.nbytes() ? x.nbytes() : 16;
  id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!b) throw std::runtime_error("buffer allocation failed");
  memset([b contents], 0, bytes);
  x.buf = (__bridge id)CFBridgingRetain(b);
  return x;
}

Cmd::Cmd() { reset(); }

Stats &stats() {
  static Stats s;
  return s;
}
void stats_reset() { stats() = Stats{}; }

Cmd::~Cmd() {
  if (enc && !ended) [enc endEncoding];
  if (enc) CFRelease((__bridge CFTypeRef)enc);
  if (cb) CFRelease((__bridge CFTypeRef)cb);
}

void Cmd::reset() {
  if (enc && !ended) [enc endEncoding];
  if (enc) CFRelease((__bridge CFTypeRef)enc);
  if (cb) CFRelease((__bridge CFTypeRef)cb);
  ++stats().cbs;
  ++stats().encs;
  // The factory methods return +0 (autoreleased); drain that reference right away so the command
  // buffer is freed as soon as this Cmd releases it, not at the next outer autorelease pool.
  @autoreleasepool {
    // Retained references: the command buffer keeps every resource it uses alive until it is
    // released, so autorelease pools can drain while GPU work is pending without freeing buffers.
    id<MTLCommandBuffer> b = [Context::get().queue commandBuffer];
    cb = (__bridge id)CFRetain((__bridge CFTypeRef)b);
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    enc = (__bridge id)CFRetain((__bridge CFTypeRef)e);
  }
  ended = false;
}

void Cmd::commit() {
  if (!ended) {
    [enc endEncoding];
    ended = true;
  }
  [cb commit];
}

void Cmd::wait() {
  ++stats().waits;
  [cb waitUntilCompleted];
  const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  stats().gpu_ms += ms;
  if (stats().stage >= 0 && stats().stage < 8) stats().stage_ms[stats().stage] += ms;
  if (cb.status == MTLCommandBufferStatusError)
    throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
}

double Cmd::gpu_ms() const { return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0; }

double now_ms() {
  return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6;
}

}  // namespace gpu
