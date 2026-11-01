// Audio front end: log-mel of Qwen3-ASR's Voxt/Swift layout (owner: encoder worker).
#include "frontend.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace fe {

namespace {

constexpr int kFreqBins = kFftSize / 2 + 1;  // 201

float HertzToMel(float frequency_hz, float linear_step_hz, float min_log_hz, float min_log_mel,
                 float log_step) {
  if (frequency_hz < min_log_hz) return frequency_hz / linear_step_hz;
  return min_log_mel + std::log(frequency_hz / min_log_hz) / log_step;
}

float MelToHertz(float mel, float linear_step_hz, float min_log_hz, float min_log_mel, float log_step) {
  if (mel < min_log_mel) return linear_step_hz * mel;
  return min_log_hz * std::exp(log_step * (mel - min_log_mel));
}

std::vector<float> build_mel_filterbank(int mel_bin_count) {
  const float linear_step_hz = 200.0f / 3.0f;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / linear_step_hz;
  const float log_step = std::log(6.4f) / 27.0f;

  std::vector<float> bin_frequencies_hz(kFreqBins);
  for (int i = 0; i < kFreqBins; ++i)
    bin_frequencies_hz[i] = static_cast<float>(i) * static_cast<float>(kSampleRate) /
                            static_cast<float>(kFftSize);
  const float mel_max = HertzToMel(static_cast<float>(kSampleRate) / 2.0f, linear_step_hz, min_log_hz,
                                   min_log_mel, log_step);
  std::vector<float> edges_hz(mel_bin_count + 2);
  for (int i = 0; i < mel_bin_count + 2; ++i)
    edges_hz[i] = MelToHertz(static_cast<float>(i) * mel_max / static_cast<float>(mel_bin_count + 1),
                             linear_step_hz, min_log_hz, min_log_mel, log_step);
  std::vector<float> filters(static_cast<size_t>(kFreqBins) * mel_bin_count, 0.0f);
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const float low = edges_hz[mel_bin];
    const float center = edges_hz[mel_bin + 1];
    const float high = edges_hz[mel_bin + 2];
    const float normalization = 2.0f / (high - low);
    for (int frequency_bin = 0; frequency_bin < kFreqBins; ++frequency_bin) {
      const float frequency_hz = bin_frequencies_hz[frequency_bin];
      float weight;
      if (low <= frequency_hz && frequency_hz < center) {
        weight = (frequency_hz - low) / (center - low);
      } else if (center <= frequency_hz && frequency_hz <= high) {
        weight = (high - frequency_hz) / (high - center);
      } else {
        weight = 0.0f;
      }
      filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin] = weight * normalization;
    }
  }
  return filters;
}

std::vector<float> build_hann() {
  std::vector<float> window(kFftSize);
  const float denominator = static_cast<float>(kFftSize);
  // Swift's Float.pi, one float step below (float)M_PI (as the reference does).
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  for (int n = 0; n < kFftSize; ++n)
    window[n] = 0.5f * (1.0f - std::cos(2.0f * pi * static_cast<float>(n) / denominator));
  return window;
}

// One-time device copies of the host tables.
struct Tables {
  gpu::Tensor filters;  // [201 * 128] f32
  gpu::Tensor hann;     // [400] f32
  Tables() {
    const auto &f = mel_filterbank();
    const auto &h = hann_window();
    auto &ctx = gpu::Context::get();
    filters = ctx.empty({(int64_t)f.size()}, gpu::DType::F32);
    hann = ctx.empty({(int64_t)h.size()}, gpu::DType::F32);
    memcpy(filters.host(), f.data(), f.size() * 4);
    memcpy(hann.host(), h.data(), h.size() * 4);
  }
};

Tables &tables() {
  static Tables t;
  return t;
}

// MLX fft.metal specialised for n = 400 (real input float, complex output float2).
id<MTLComputePipelineState> fft_pipeline() {
  static id<MTLComputePipelineState> p = nil;
  static std::once_flag once;
  std::call_once(once, [] {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    bool no = false;
    int elems_per_thread = 5;   // compute_elems_per_thread for {8,5,2}
    int n = 400;
    int rader_m = 400;          // n / rader_n
    int zero = 0, one = 1, two = 2;
    [cv setConstantValue:&no type:MTLDataTypeBool atIndex:0];   // inverse
    [cv setConstantValue:&no type:MTLDataTypeBool atIndex:1];   // power_of_2
    [cv setConstantValue:&elems_per_thread type:MTLDataTypeInt atIndex:2];
    [cv setConstantValue:&rader_m type:MTLDataTypeInt atIndex:3];
    for (int i = 4; i <= 21; ++i) [cv setConstantValue:&zero type:MTLDataTypeInt atIndex:i];
    [cv setConstantValue:&one type:MTLDataTypeInt atIndex:6];   // radix_8_steps
    [cv setConstantValue:&two type:MTLDataTypeInt atIndex:9];   // radix_5_steps
    [cv setConstantValue:&one type:MTLDataTypeInt atIndex:12];  // radix_2_steps
    p = gpu::Context::get().pipeline("enc_fft_400", "n400", cv);
  });
  return p;
}

}  // namespace

const std::vector<float> &mel_filterbank() {
  static const std::vector<float> f = build_mel_filterbank(kMelBins);
  return f;
}

const std::vector<float> &hann_window() {
  static const std::vector<float> w = build_hann();
  return w;
}

int frame_count(size_t n_samples) {
  const size_t n = std::max<size_t>(n_samples, kFftSize);  // short cuts are zero-filled
  return 1 + static_cast<int>(n / kHopLength);
}

MelResult log_mel(const float *samples, size_t n_samples, int f0, int f1, float clamp_max) {
  if (f1 <= f0) throw std::runtime_error("log_mel: empty frame range");
  if (f0 < 0 || f1 > frame_count(n_samples)) throw std::runtime_error("log_mel: frame range out of bounds");
  const size_t n = std::max<size_t>(n_samples, kFftSize);
  const int frames = f1 - f0;
  auto &ctx = gpu::Context::get();

  // Padded samples [160*f0, 160*f1 + 400).
  const int p0 = kHopLength * f0;
  const int plen = kHopLength * frames + kFftSize;
  gpu::Tensor audio = ctx.empty({(int64_t)n}, gpu::DType::F32);
  memcpy(audio.host(), samples, n_samples * sizeof(float));
  gpu::Tensor pad = ctx.empty({plen}, gpu::DType::F32);
  gpu::Tensor wframes = ctx.empty({(int64_t)frames * kFftSize}, gpu::DType::F32);
  gpu::Tensor spec = ctx.empty({(int64_t)frames * kFreqBins, 2}, gpu::DType::F32);
  gpu::Tensor raw = ctx.empty({(int64_t)frames * kMelBins}, gpu::DType::F32);
  gpu::Tensor fmax = ctx.empty({frames}, gpu::DType::F32);
  gpu::Tensor maxbuf = ctx.empty({1}, gpu::DType::F32);
  gpu::Tensor mel = ctx.empty({kMelBins, frames}, gpu::DType::F32);
  gpu::Tensor clamp = ctx.empty({1}, gpu::DType::F32);

  {
    gpu::Cmd c;
    c.use("enc_pad");
    c.set(audio, 0);
    c.set(pad, 1);
    uint32_t nn = (uint32_t)n;
    c.set_value(nn, 2);
    c.set_value(p0, 3);
    uint32_t len = (uint32_t)plen;
    c.set_value(len, 4);
    c.dispatch_threads(MTLSizeMake(plen, 1, 1), MTLSizeMake(256, 1, 1));

    c.use("enc_window");
    c.set(pad, 0);
    c.set(tables().hann, 1);
    c.set(wframes, 2);
    uint32_t fc = (uint32_t)frames;
    c.set_value(fc, 3);
    c.dispatch_threads(MTLSizeMake(kFftSize, frames, 1), MTLSizeMake(kFftSize, 1, 1));

    // MLX fft kernel: 2 real FFTs per threadgroup, 80 threads of 5 elements each.
    c.use(fft_pipeline());
    c.set(wframes, 0);
    c.set(spec, 1);
    int nfft = kFftSize;
    c.set_value(nfft, 2);
    c.set_value((int)frames, 3);
    c.dispatch_threads(MTLSizeMake((frames + 1) / 2, 1, 80), MTLSizeMake(1, 1, 80));

    c.use("enc_mel_log");
    c.set(spec, 0);
    c.set(tables().filters, 1);
    c.set(raw, 2);
    c.set(fmax, 3);
    c.set_value(fc, 4);
    c.dispatch_groups(MTLSizeMake(frames, 1, 1), MTLSizeMake(128, 1, 1));

    c.use("enc_max_reduce");
    c.set(fmax, 0);
    c.set(maxbuf, 1);
    c.set_value(fc, 2);
    c.dispatch_groups(MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
    c.commit_and_wait();
  }

  const float raw_max = maxbuf.ptr<float>()[0];
  const float use_max = std::isfinite(clamp_max) ? clamp_max : raw_max;
  clamp.ptr<float>()[0] = use_max;
  {
    gpu::Cmd c;
    c.use("enc_mel_finalize");
    c.set(raw, 0);
    c.set(clamp, 1);
    c.set(mel, 2);
    uint32_t fc = (uint32_t)frames;
    c.set_value(fc, 3);
    c.dispatch_threads(MTLSizeMake(frames, kMelBins, 1), MTLSizeMake(32, 4, 1));
    c.commit_and_wait();
  }
  MelResult r;
  r.mel = mel;
  r.f0 = f0;
  r.frames = frames;
  r.raw_max = raw_max;
  return r;
}

MelResult log_mel_all(const float *samples, size_t n) {
  return log_mel(samples, n, 0, frame_count(n), -INFINITY);
}

}  // namespace fe
