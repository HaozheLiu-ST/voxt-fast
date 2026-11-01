// Qwen2 byte-level BPE tokenizer, ported from SGLang-Omni common/tokenizer.cpp
// (Apache-2.0). The pre-tokenizer split uses NSRegularExpression (ICU) with
// the checkpoint's own pattern; NFC and UTF-8 lossy decoding use CoreFoundation
// system facilities instead of generated Unicode tables.
#import <Foundation/Foundation.h>

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "json.h"
#include "tokenizer.h"

namespace {

// Qwen2's pre-tokenization split, verbatim from the checkpoint's tokenizer.
constexpr const char *kQwen2SplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N})"
    R"(| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void AppendUtf8(uint32_t code_point, std::string &out) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

// GPT-2's byte-to-unicode table; printable bytes map to themselves, the rest to
// code points from 256 upward.
struct ByteLevelTables {
  std::array<std::string, 256> byte_to_text;
  std::unordered_map<uint32_t, uint8_t> code_point_to_byte;

  ByteLevelTables() {
    std::array<uint32_t, 256> code_points{};
    std::array<bool, 256> printable{};
    for (int b = '!'; b <= '~'; ++b) printable[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      code_points[b] = printable[b] ? static_cast<uint32_t>(b) : next++;
    }
    for (int b = 0; b < 256; ++b) {
      AppendUtf8(code_points[b], byte_to_text[b]);
      code_point_to_byte[code_points[b]] = static_cast<uint8_t>(b);
    }
  }
};

const ByteLevelTables &Tables() {
  static const ByteLevelTables tables;
  return tables;
}

// Assumes valid UTF-8; advances offset.
uint32_t NextCodePoint(const std::string &text, size_t &offset) {
  const auto byte = [&](size_t i) { return static_cast<uint8_t>(text[i]); };
  const uint8_t lead = byte(offset);
  if (lead < 0x80) {
    offset += 1;
    return lead;
  } else if ((lead >> 5) == 0x6) {
    const uint32_t cp = ((lead & 0x1F) << 6) | (byte(offset + 1) & 0x3F);
    offset += 2;
    return cp;
  } else if ((lead >> 4) == 0xE) {
    const uint32_t cp = ((lead & 0x0F) << 12) |
                        ((byte(offset + 1) & 0x3F) << 6) |
                        (byte(offset + 2) & 0x3F);
    offset += 3;
    return cp;
  } else {
    const uint32_t cp = ((lead & 0x07) << 18) | ((byte(offset + 1) & 0x3F) << 12) |
                        ((byte(offset + 2) & 0x3F) << 6) | (byte(offset + 3) & 0x3F);
    offset += 4;
    return cp;
  }
}

// NFC, as the tokenizer's normalizer applies to ordinary text.
std::string NormalizeNfc(const std::string &text) {
  if (std::all_of(text.begin(), text.end(),
                  [](char c) { return static_cast<uint8_t>(c) < 0x80; })) {
    return text;
  }
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (source == nullptr) return text;
  CFMutableStringRef normalized =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  CFStringNormalize(normalized, kCFStringNormalizationFormC);
  const CFIndex length = CFStringGetLength(normalized);
  CFIndex byte_count = 0;
  CFStringGetBytes(normalized, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, nullptr, 0, &byte_count);
  std::string out(static_cast<size_t>(byte_count), '\0');
  CFStringGetBytes(normalized, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, reinterpret_cast<UInt8 *>(out.data()), byte_count,
                   nullptr);
  CFRelease(normalized);
  return out;
}

// Matches Rust's String::from_utf8_lossy, one U+FFFD per maximal invalid
// subpart.
std::string FromUtf8Lossy(const std::string &bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  const auto byte = [&](size_t k) { return static_cast<uint8_t>(bytes[k]); };
  while (i < bytes.size()) {
    const uint8_t lead = byte(i);
    if (lead < 0x80) {
      out.push_back(static_cast<char>(lead));
      ++i;
      continue;
    }
    size_t need = 0;
    uint8_t low = 0x80;
    uint8_t high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
      need = 1;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      need = 2;
      low = lead == 0xE0 ? 0xA0 : 0x80;
      high = lead == 0xED ? 0x9F : 0xBF;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      need = 3;
      low = lead == 0xF0 ? 0x90 : 0x80;
      high = lead == 0xF4 ? 0x8F : 0xBF;
    } else {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    size_t consumed = 1;
    bool valid = true;
    for (size_t k = 0; k < need; ++k) {
      if (i + consumed >= bytes.size()) {
        valid = false;
        break;
      }
      const uint8_t continuation = byte(i + consumed);
      const uint8_t allowed_low = k == 0 ? low : 0x80;
      const uint8_t allowed_high = k == 0 ? high : 0xBF;
      if (continuation < allowed_low || continuation > allowed_high) {
        valid = false;
        break;
      }
      ++consumed;
    }
    if (valid) {
      out.append(bytes, i, consumed);
    } else {
      out += "\xEF\xBF\xBD";
    }
    i += consumed;
  }
  return out;
}

// UTF-16 unit index -> UTF-8 byte offset (size units + 1). Empty when the text
// is not valid UTF-8, in which case the caller encodes it as one piece.
std::vector<size_t> Utf16ToByteOffsets(const std::string &utf8) {
  std::vector<size_t> offsets;
  offsets.reserve(utf8.size() + 1);
  size_t i = 0;
  while (i < utf8.size()) {
    const uint8_t lead = static_cast<uint8_t>(utf8[i]);
    size_t length = 0;
    if (lead < 0x80) {
      length = 1;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      length = i + 1 < utf8.size() ? 2 : 0;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = i + 2 < utf8.size() ? 3 : 0;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = i + 3 < utf8.size() ? 4 : 0;
    }
    if (length == 0) return {};
    for (size_t k = 0; k < length; ++k) {
      if ((static_cast<uint8_t>(utf8[i + k]) & 0xC0) != 0x80 && k > 0) return {};
    }
    offsets.push_back(i);
    if (length == 4) offsets.push_back(i);  // surrogate pair: two UTF-16 units
    i += length;
  }
  offsets.push_back(utf8.size());
  return offsets;
}

}  // namespace

struct Tokenizer::Impl {
  struct AddedToken {
    std::string content;
    int id = -1;
    bool special = false;
  };

  std::unordered_map<std::string, int> vocabulary;
  std::unordered_map<std::string, int> merge_ranks;  // "left right" -> rank
  std::vector<AddedToken> added_tokens;
  std::unordered_map<int, size_t> added_index_by_id;
  std::vector<std::string> tokens_by_id;
  NSRegularExpression *split_pattern = nil;

  void EncodeOrdinary(const std::string &raw_text, std::vector<int> &ids) const;
  void EncodeWord(const std::string &byte_level_word, std::vector<int> &ids) const;
  void EncodePiece(const std::string &text, size_t begin, size_t end,
                   std::vector<int> &ids) const;
};

Tokenizer::Tokenizer(const std::filesystem::path &model_directory)
    : impl_(std::make_unique<Impl>()) {
  if (!std::filesystem::exists(model_directory / "vocab.json")) {
    throw std::runtime_error("tokenizer needs vocab.json in " +
                             model_directory.string());
  }
  const json::Value vocabulary =
      json::parse(ReadFile(model_directory / "vocab.json"));
  for (const auto &[token, id] : vocabulary.obj) {
    impl_->vocabulary.emplace(token, static_cast<int>(id.as_int()));
  }
  std::istringstream merges(ReadFile(model_directory / "merges.txt"));
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (line.rfind("#version", 0) == 0 || line.empty()) continue;
    impl_->merge_ranks.emplace(line, rank++);
  }
  const json::Value config =
      json::parse(ReadFile(model_directory / "tokenizer_config.json"));
  const json::Value *decoder = config.find("added_tokens_decoder");
  if (decoder == nullptr) {
    throw std::runtime_error("tokenizer_config.json has no added_tokens_decoder");
  }
  for (const auto &[id_text, token] : decoder->obj) {
    const int id = std::stoi(id_text);
    impl_->added_index_by_id[id] = impl_->added_tokens.size();
    impl_->added_tokens.push_back({token["content"].as_str(), id,
                                   token["special"].b});
  }
  size_t largest_id = 0;
  for (const auto &[token, id] : impl_->vocabulary) {
    largest_id = std::max(largest_id, static_cast<size_t>(id));
  }
  for (const auto &added : impl_->added_tokens) {
    largest_id = std::max(largest_id, static_cast<size_t>(added.id));
  }
  impl_->tokens_by_id.resize(largest_id + 1);
  for (const auto &[token, id] : impl_->vocabulary) {
    impl_->tokens_by_id[id] = token;
  }
  NSError *error = nil;
  impl_->split_pattern = [NSRegularExpression
      regularExpressionWithPattern:[NSString stringWithUTF8String:kQwen2SplitPattern]
                           options:0
                             error:&error];
  if (impl_->split_pattern == nil) {
    throw std::runtime_error(std::string("cannot compile the tokenizer split "
                                         "pattern: ") +
                             error.localizedDescription.UTF8String);
  }
}

Tokenizer::~Tokenizer() = default;

int Tokenizer::AddedTokenId(const std::string &content) const {
  for (const auto &added : impl_->added_tokens) {
    if (added.content == content) return added.id;
  }
  return -1;
}

int Tokenizer::VocabSize() const {
  return static_cast<int>(impl_->tokens_by_id.size());
}

std::vector<int> Tokenizer::Encode(const std::string &text) const {
  // Added tokens match on the raw text first (leftmost, longest); only the text
  // between them is normalized and BPE-encoded.
  std::vector<int> ids;
  size_t ordinary_start = 0;
  size_t position = 0;
  while (position < text.size()) {
    const Impl::AddedToken *match = nullptr;
    for (const auto &added : impl_->added_tokens) {
      if (text.compare(position, added.content.size(), added.content) == 0 &&
          (match == nullptr || added.content.size() > match->content.size())) {
        match = &added;
      }
    }
    if (match != nullptr) {
      impl_->EncodeOrdinary(
          text.substr(ordinary_start, position - ordinary_start), ids);
      ids.push_back(match->id);
      position += match->content.size();
      ordinary_start = position;
    } else {
      ++position;
    }
  }
  impl_->EncodeOrdinary(text.substr(ordinary_start), ids);
  return ids;
}

void Tokenizer::Impl::EncodePiece(const std::string &text, size_t begin,
                                  size_t end, std::vector<int> &ids) const {
  std::string byte_level;
  byte_level.reserve((end - begin) * 2);
  for (size_t i = begin; i < end; ++i) {
    byte_level += Tables().byte_to_text[static_cast<uint8_t>(text[i])];
  }
  EncodeWord(byte_level, ids);
}

void Tokenizer::Impl::EncodeOrdinary(const std::string &raw_text,
                                     std::vector<int> &ids) const {
  if (raw_text.empty()) return;
  const std::string text = NormalizeNfc(raw_text);
  NSString *subject = [[NSString alloc] initWithBytes:text.data()
                                              length:text.size()
                                            encoding:NSUTF8StringEncoding];
  const std::vector<size_t> offsets = Utf16ToByteOffsets(text);
  if (subject == nil || split_pattern == nil || offsets.empty()) {
    // Invalid UTF-8 (never produced by Decode): encode the whole text as one
    // piece, which is what the reference does when its split match fails.
    EncodePiece(text, 0, text.size(), ids);
    return;
  }
  NSArray<NSTextCheckingResult *> *matches = [split_pattern
      matchesInString:subject
              options:0
                range:NSMakeRange(0, subject.length)];
  size_t previous_end = 0;
  for (NSTextCheckingResult *match in matches) {
    const NSUInteger begin = match.range.location;
    const NSUInteger end = match.range.location + match.range.length;
    if (end >= offsets.size()) continue;
    const size_t match_begin = offsets[begin];
    const size_t match_end = offsets[end];
    if (match_end == match_begin) continue;
    if (match_begin > previous_end) {
      EncodePiece(text, previous_end, match_begin, ids);
    }
    EncodePiece(text, match_begin, match_end, ids);
    previous_end = match_end;
  }
  if (previous_end < text.size()) {
    EncodePiece(text, previous_end, text.size(), ids);
  }
}

void Tokenizer::Impl::EncodeWord(const std::string &byte_level_word,
                                 std::vector<int> &ids) const {
  std::vector<std::string> symbols;
  size_t offset = 0;
  while (offset < byte_level_word.size()) {
    const size_t start = offset;
    NextCodePoint(byte_level_word, offset);
    symbols.push_back(byte_level_word.substr(start, offset - start));
  }
  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_index = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto rank = merge_ranks.find(symbols[i] + " " + symbols[i + 1]);
      if (rank != merge_ranks.end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_index = i;
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;
    const std::string left = symbols[best_index];
    const std::string right = symbols[best_index + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (i + 1 < symbols.size() && symbols[i] == left &&
          symbols[i + 1] == right) {
        merged.push_back(left + right);
        ++i;
      } else {
        merged.push_back(symbols[i]);
      }
    }
    symbols.swap(merged);
  }
  for (const auto &symbol : symbols) {
    const auto id = vocabulary.find(symbol);
    if (id != vocabulary.end()) ids.push_back(id->second);
  }
}

std::string Tokenizer::Decode(const std::vector<int> &ids,
                              bool skip_special_tokens) const {
  std::string bytes;
  for (const int id : ids) {
    const auto added = impl_->added_index_by_id.find(id);
    std::string token;
    if (added != impl_->added_index_by_id.end()) {
      const Impl::AddedToken &added_token = impl_->added_tokens[added->second];
      if (skip_special_tokens && added_token.special) continue;
      token = added_token.content;
    } else if (id >= 0 && static_cast<size_t>(id) < impl_->tokens_by_id.size()) {
      token = impl_->tokens_by_id[id];
    } else {
      continue;
    }
    // A token whose characters do not all map back to bytes keeps its own
    // UTF-8.
    std::string token_bytes;
    bool all_mapped = true;
    size_t offset = 0;
    while (offset < token.size()) {
      const uint32_t code_point = NextCodePoint(token, offset);
      const auto mapped = Tables().code_point_to_byte.find(code_point);
      if (mapped == Tables().code_point_to_byte.end()) {
        all_mapped = false;
        break;
      }
      token_bytes.push_back(static_cast<char>(mapped->second));
    }
    bytes += all_mapped ? token_bytes : token;
  }
  return FromUtf8Lossy(bytes);
}
