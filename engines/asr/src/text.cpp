// Model-independent text helpers, ported from SGLang-Omni common/text.cpp and
// qwen3_asr/qwen3_asr_transcriber.cpp (Apache-2.0). Behaviour is kept
// statement-for-statement.
#include "text.h"

#include <algorithm>
#include <cctype>
#include <set>

#include "prompt.h"
#include "tokenizer.h"

namespace host {

namespace {

constexpr size_t kTokenLoopWindow = 24;
constexpr size_t kTokenLoopMaxDistinct = 3;
constexpr const char *kAsrTextMarker = "<asr_text>";

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

std::optional<size_t> FindSubsequence(const std::vector<int> &values,
                                      const std::vector<int> &pattern) {
  if (pattern.size() > values.size()) return std::nullopt;
  for (size_t start = 0; start + pattern.size() <= values.size(); ++start) {
    if (std::equal(pattern.begin(), pattern.end(), values.begin() + start)) {
      return start;
    }
  }
  return std::nullopt;
}

}  // namespace

bool IsUnicodeSpace(uint32_t code_point) {
  return code_point == ' ' || (code_point >= 0x09 && code_point <= 0x0D) ||
         (code_point >= 0x1C && code_point <= 0x1F) || code_point == 0x85 ||
         code_point == 0xA0 || code_point == 0x1680 ||
         (code_point >= 0x2000 && code_point <= 0x200A) ||
         code_point == 0x2028 || code_point == 0x2029 || code_point == 0x202F ||
         code_point == 0x205F || code_point == 0x3000;
}

std::vector<uint32_t> CodePoints(const std::string &text) {
  std::vector<uint32_t> code_points;
  size_t i = 0;
  while (i < text.size()) {
    const auto byte = [&](size_t k) { return static_cast<uint8_t>(text[k]); };
    const uint8_t lead = byte(i);
    uint32_t cp = lead;
    size_t length = 1;
    if (lead >= 0xF0 && i + 3 < text.size()) {
      cp = ((lead & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12) |
           ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
      length = 4;
    } else if (lead >= 0xE0 && i + 2 < text.size()) {
      cp = ((lead & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) |
           (byte(i + 2) & 0x3F);
      length = 3;
    } else if (lead >= 0xC0 && i + 1 < text.size()) {
      cp = ((lead & 0x1F) << 6) | (byte(i + 1) & 0x3F);
      length = 2;
    }
    code_points.push_back(cp);
    i += length;
  }
  return code_points;
}

std::string StripUnicodeWhitespace(const std::string &text) {
  const std::vector<uint32_t> code_points = CodePoints(text);
  size_t begin_cp = 0;
  size_t end_cp = code_points.size();
  while (begin_cp < end_cp && IsUnicodeSpace(code_points[begin_cp])) ++begin_cp;
  while (end_cp > begin_cp && IsUnicodeSpace(code_points[end_cp - 1])) --end_cp;
  size_t byte = 0;
  size_t begin_byte = 0;
  size_t end_byte = 0;
  for (size_t cp = 0; cp <= code_points.size(); ++cp) {
    if (cp == begin_cp) begin_byte = byte;
    if (cp == end_cp) end_byte = byte;
    if (cp == code_points.size()) break;
    const uint8_t lead = static_cast<uint8_t>(text[byte]);
    byte += lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
  }
  return text.substr(begin_byte, end_byte - begin_byte);
}

bool IsTokenLoop(const std::vector<int> &output_ids) {
  if (output_ids.size() < kTokenLoopWindow) return false;
  const std::set<int> distinct(output_ids.end() - kTokenLoopWindow,
                               output_ids.end());
  return distinct.size() <= kTokenLoopMaxDistinct;
}

bool IsStopToken(int token_id, const StopRules &rules) {
  return token_id == rules.im_end_id ||
         (rules.stop_at_end_of_text && token_id == rules.end_of_text_id);
}

int GeneratedTokenCount(const std::vector<int> &output_ids,
                        const StopRules &rules) {
  const bool ended_on_stop =
      !output_ids.empty() && IsStopToken(output_ids.back(), rules);
  return static_cast<int>(output_ids.size()) - (ended_on_stop ? 1 : 0);
}

SplitResult SplitOutput(const Tokenizer &tokenizer,
                        const std::vector<int> &output_ids,
                        const std::string &requested_language) {
  const std::vector<int> asr_text_ids = tokenizer.Encode(kAsrTextMarker);
  const std::optional<size_t> marker =
      FindSubsequence(output_ids, asr_text_ids);
  std::optional<std::string> detected;
  if (requested_language.empty() && marker.has_value()) {
    const std::string prefix = Strip(tokenizer.Decode(
        std::vector<int>(output_ids.begin(), output_ids.begin() + *marker),
        true));
    const size_t separator = prefix.find(' ');
    if (separator != std::string::npos &&
        Lowercase(prefix.substr(0, separator)) == "language") {
      const std::string value = Strip(prefix.substr(separator + 1));
      if (!value.empty() && Lowercase(value) != "none") detected = value;
    } else if (!prefix.empty()) {
      detected = prefix;
    }
  }
  const std::vector<int> transcript_ids =
      marker.has_value()
          ? std::vector<int>(output_ids.begin() + *marker + asr_text_ids.size(),
                             output_ids.end())
          : output_ids;
  SplitResult result;
  result.text = tokenizer.Decode(transcript_ids, true);
  std::string language = NormalizeLanguage(requested_language);
  if (language.empty() && detected.has_value()) {
    language = NormalizeLanguage(*detected);
  }
  if (!language.empty()) result.language = language;
  return result;
}

}  // namespace host
