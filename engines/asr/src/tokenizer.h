// Qwen2 byte-level BPE tokenizer (vocab.json + merges.txt + tokenizer_config.json
// added tokens), ported from SGLang-Omni common/tokenizer.cpp (Apache-2.0).
// No PCRE2 at run time: the Qwen2 pre-tokenizer split uses NSRegularExpression
// (ICU), which supports \p{L}, \p{N}, \s and the (?!\S) lookahead.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path &model_directory);
  ~Tokenizer();
  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;

  // Raw text -> ids. Added tokens match first (leftmost, longest), the text
  // between them is NFC-normalized and byte-level BPE encoded.
  std::vector<int> Encode(const std::string &text) const;

  // Lossy UTF-8 decoding: invalid bytes become U+FFFD, as Hugging Face does.
  // skip_special_tokens drops added tokens whose "special" flag is true.
  std::string Decode(const std::vector<int> &ids, bool skip_special_tokens) const;

  // -1 when the token is not an added token.
  int AddedTokenId(const std::string &content) const;

  int VocabSize() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
