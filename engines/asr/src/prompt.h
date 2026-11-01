// Qwen3-ASR prompt layout (voxt_swift) and audio-token counts.
// Ported from SGLang-Omni qwen3_asr_transcriber.cpp and qwen3_audio.cpp
// (Apache-2.0).
#pragma once

#include <string>
#include <vector>

class Tokenizer;

namespace host {

// Qwen3-ASR encodes mel frames in chunks of 100 frames, 13 audio tokens each.
inline constexpr int kChunkFrameCount = 100;
inline constexpr int kChunkTokenCount = 13;

enum class AudioLayout { kReference, kVoxtSwift };

int ConvOutputFrames(int frame_count);
int ReferenceTokenCount(int frame_count);
int SwiftTokenCount(int frame_count);
int TokenCount(int frame_count, AudioLayout layout);

struct PromptOptions {
  std::string language = "English";  // empty = do not emit a language prefix
  std::string context;               // system prompt (empty for this task)
  std::vector<int> prefix_token_ids;
  AudioLayout layout = AudioLayout::kVoxtSwift;
};

struct Prompt {
  std::vector<int> ids;
  int audio_pad_start = -1;  // index of the first <|audio_pad|>, -1 if none
  int audio_token_count = 0;
};

// Exactly Qwen3ASRTranscriber::PromptIds for the given audio_pad count.
Prompt PromptIds(const Tokenizer &tokenizer, int audio_token_count,
                 const PromptOptions &options);

// Convenience: audio_pad count = TokenCount(mel_frame_count, options.layout).
Prompt BuildPrompt(const Tokenizer &tokenizer, int mel_frame_count,
                   const PromptOptions &options);

// "" when the language is empty; otherwise the reference's normalized name
// ("English", "Chinese", ...). Unknown non-empty names are returned stripped.
std::string NormalizeLanguage(const std::string &language);

}  // namespace host
