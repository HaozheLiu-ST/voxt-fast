// Model-independent text helpers: UTF-8 code points, Unicode whitespace,
// greedy token-loop detection, stop rules and output splitting.
// Ported from SGLang-Omni common/text.cpp and qwen3_asr_transcriber.cpp
// (Apache-2.0).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class Tokenizer;

namespace host {

bool IsUnicodeSpace(uint32_t code_point);
std::vector<uint32_t> CodePoints(const std::string &text);
// Strips every Unicode space from both ends.
std::string StripUnicodeWhitespace(const std::string &text);

// Greedy loop guard: the last 24 tokens use at most 3 distinct ids.
bool IsTokenLoop(const std::vector<int> &output_ids);

struct StopRules {
  int im_end_id = -1;
  int end_of_text_id = -1;
  bool stop_at_end_of_text = true;
  bool stop_on_token_loop = true;
};
bool IsStopToken(int token_id, const StopRules &rules);

struct SplitResult {
  std::string text;
  std::optional<std::string> language;
};

// Strips "language X<asr_text>" (if present), decodes the transcript with
// skip_special_tokens=true, and resolves the language exactly as
// Qwen3ASRTranscriber::SplitOutput. requested_language empty = unset.
SplitResult SplitOutput(const Tokenizer &tokenizer,
                        const std::vector<int> &output_ids,
                        const std::string &requested_language);

// Transcribe semantics: the final stop token is not counted.
int GeneratedTokenCount(const std::vector<int> &output_ids,
                        const StopRules &rules);

}  // namespace host
