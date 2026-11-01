// Audio front end: Whisper-style log-mel of Qwen3-ASR's Voxt/Swift layout (owner: encoder worker).
//
// Reference: qwen3_asr/qwen3_audio.cpp LogMel(samples, AudioLayout::kVoxtSwift) with
// common/audio.cpp (reflect pad 200, periodic Hann 400, rfft n=400 hop 160, power,
// mel filterbank [201 x 128], log10 clamp 1e-10, dynamic range 8 below the max, (x+4)/4).
// Layout of the result: [mel_bin, frame] row-major (mel bin major), exactly as the reference
// hands it to EncodeAudio.
#pragma once

#include <cstddef>
#include <vector>

#include "gpu.h"

namespace fe {

constexpr int kSampleRate = 16000;
constexpr int kHopLength = 160;
constexpr int kFftSize = 400;
constexpr int kMelBins = 128;
constexpr float kLogMelFloor = 1e-10f;
constexpr float kLogMelDynamicRange = 8.0f;

// Number of mel frames the Swift layout keeps for `n` samples: 1 + n / 160 (n < 400 is
// zero-filled to one window, as the reference does for very short realtime cuts).
int frame_count(size_t n_samples);

struct MelResult {
  gpu::Tensor mel;   // [kMelBins, frames] float32, frames = f1 - f0
  int f0 = 0;
  int frames = 0;
  float raw_max = 0;  // max of the unclamped log10 spectrum over [f0, f1)
};

// Mel frames [f0, f1) of a recording of `n` samples of mono float32 audio at 16 kHz.
// `clamp_max` is the dynamic-range reference (the reference uses the max over the whole
// recording); pass -INFINITY to use the max over [f0, f1) itself.
MelResult log_mel(const float *samples, size_t n, int f0, int f1, float clamp_max = -INFINITY);

// Convenience: all frames of a recording, dynamic range taken over the whole recording.
MelResult log_mel_all(const float *samples, size_t n);

// The reference's host-side tables (float32, built in the same order as the reference).
const std::vector<float> &mel_filterbank();  // [201 * 128], index freq_bin * 128 + mel_bin
const std::vector<float> &hann_window();     // [400]

}  // namespace fe
