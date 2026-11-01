// Qwen3-ASR prompt layout (voxt_swift) and audio-token counts, ported from
// SGLang-Omni qwen3_asr/qwen3_asr_transcriber.cpp and qwen3_asr/qwen3_audio.cpp
// (Apache-2.0).
#include "prompt.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>

#include "tokenizer.h"

namespace host {

namespace {

constexpr const char *kAsrTextMarker = "<asr_text>";
constexpr const char *kAudioPad = "<|audio_pad|>";

const std::map<std::string, std::string> &LanguageCodeToName() {
  static const std::map<std::string, std::string> table = {
      {"ar", "Arabic"},     {"yue", "Cantonese"}, {"zh", "Chinese"},
      {"cs", "Czech"},      {"da", "Danish"},     {"nl", "Dutch"},
      {"en", "English"},    {"fil", "Filipino"},  {"fi", "Finnish"},
      {"fr", "French"},     {"de", "German"},     {"el", "Greek"},
      {"hi", "Hindi"},      {"hu", "Hungarian"},  {"id", "Indonesian"},
      {"it", "Italian"},    {"ja", "Japanese"},   {"ko", "Korean"},
      {"mk", "Macedonian"}, {"ms", "Malay"},      {"fa", "Persian"},
      {"pl", "Polish"},     {"pt", "Portuguese"}, {"ro", "Romanian"},
      {"ru", "Russian"},    {"es", "Spanish"},    {"sv", "Swedish"},
      {"th", "Thai"},       {"tr", "Turkish"},    {"vi", "Vietnamese"},
  };
  return table;
}

std::string Strip(const std::string &text) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && is_space(text[begin])) ++begin;
  while (end > begin && is_space(text[end - 1])) --end;
  return text.substr(begin, end - begin);
}

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

// Python floor division; the remainder formula can pass negative operands.
int FloorDivide(int numerator, int denominator) {
  const int quotient = numerator / denominator;
  return (numerator % denominator != 0 && ((numerator < 0) != (denominator < 0)))
             ? quotient - 1
             : quotient;
}

int RemainderTokens(int frame_count) {
  const int remainder_frames = frame_count % kChunkFrameCount;
  const int after_first_conv = FloorDivide(remainder_frames - 1, 2) + 1;
  return FloorDivide(FloorDivide(after_first_conv - 1, 2) + 1 - 1, 2) + 1;
}

}  // namespace

std::string NormalizeLanguage(const std::string &language) {
  const std::string stripped = Strip(language);
  const std::string normalized = Lowercase(stripped);
  if (stripped.empty()) return "";
  if (normalized == "cn" || normalized.rfind("zh-", 0) == 0 ||
      normalized.rfind("zh_", 0) == 0) {
    return "Chinese";
  }
  const auto code = LanguageCodeToName().find(normalized);
  if (code != LanguageCodeToName().end()) return code->second;
  for (const auto &[unused_code, name] : LanguageCodeToName()) {
    if (Lowercase(name) == normalized) return name;
  }
  return stripped;
}

int ConvOutputFrames(int frame_count) {
  for (int i = 0; i < 3; ++i) frame_count = (frame_count - 1) / 2 + 1;
  return frame_count;
}

int ReferenceTokenCount(int frame_count) {
  return RemainderTokens(frame_count) +
         (frame_count / kChunkFrameCount) * kChunkTokenCount;
}

int SwiftTokenCount(int frame_count) {
  const float chunks =
      static_cast<float>(frame_count) / static_cast<float>(kChunkFrameCount);
  return static_cast<int>(
      std::trunc(static_cast<float>(RemainderTokens(frame_count)) +
                 chunks * static_cast<float>(kChunkTokenCount)));
}

int TokenCount(int frame_count, AudioLayout layout) {
  return layout == AudioLayout::kReference ? ReferenceTokenCount(frame_count)
                                           : SwiftTokenCount(frame_count);
}

Prompt PromptIds(const Tokenizer &tokenizer, int audio_token_count,
                 const PromptOptions &options) {
  std::string prompt = "<|im_start|>system\n" + Strip(options.context) +
                       "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
  for (int i = 0; i < audio_token_count; ++i) prompt += kAudioPad;
  prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
  const std::string language = NormalizeLanguage(options.language);
  if (!language.empty()) {
    prompt += "language " + language + kAsrTextMarker;
  }
  Prompt result;
  result.ids = tokenizer.Encode(prompt);
  result.ids.insert(result.ids.end(), options.prefix_token_ids.begin(),
                    options.prefix_token_ids.end());
  result.audio_token_count = audio_token_count;
  const int audio_pad_id = tokenizer.AddedTokenId(kAudioPad);
  const auto first =
      std::find(result.ids.begin(), result.ids.end(), audio_pad_id);
  result.audio_pad_start =
      first == result.ids.end() ? -1 : static_cast<int>(first - result.ids.begin());
  return result;
}

Prompt BuildPrompt(const Tokenizer &tokenizer, int mel_frame_count,
                   const PromptOptions &options) {
  return PromptIds(tokenizer, TokenCount(mel_frame_count, options.layout),
                   options);
}

}  // namespace host
