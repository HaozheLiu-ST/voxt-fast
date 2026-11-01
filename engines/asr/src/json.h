// Minimal JSON reader/writer (enough for safetensors headers, config/vocab files and the stdin protocol).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

struct Value {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;  // insertion order kept

  const Value *find(const std::string &k) const {
    for (auto &kv : obj)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  const Value &operator[](const std::string &k) const {
    auto *v = find(k);
    if (!v) throw std::runtime_error("json: missing key " + k);
    return *v;
  }
  const Value &operator[](size_t i) const { return arr.at(i); }
  int64_t as_int() const { return (int64_t)num; }
  double as_num() const { return num; }
  const std::string &as_str() const { return str; }
};

class Parser {
 public:
  Parser(const char *p, const char *e) : p_(p), e_(e) {}
  Value parse() {
    Value v = value();
    return v;
  }

 private:
  const char *p_, *e_;
  void ws() {
    while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }
  [[noreturn]] void fail(const char *m) { throw std::runtime_error(std::string("json parse error: ") + m); }
  static void utf8(std::string &o, uint32_t c) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
  }
  uint32_t hex4() {
    if (e_ - p_ < 4) fail("short \\u");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = *p_++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad hex");
    }
    return v;
  }
  std::string string() {
    if (*p_ != '"') fail("expected string");
    ++p_;
    std::string o;
    while (p_ < e_ && *p_ != '"') {
      char c = *p_++;
      if (c != '\\') { o += c; continue; }
      if (p_ >= e_) fail("bad escape");
      char x = *p_++;
      switch (x) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t c1 = hex4();
          if (c1 >= 0xD800 && c1 < 0xDC00 && e_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
            p_ += 2;
            uint32_t c2 = hex4();
            c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
          }
          utf8(o, c1);
          break;
        }
        default: fail("bad escape");
      }
    }
    if (p_ >= e_) fail("unterminated string");
    ++p_;
    return o;
  }
  Value value() {
    ws();
    if (p_ >= e_) fail("unexpected end");
    Value v;
    char c = *p_;
    if (c == '{') {
      v.kind = Value::Object;
      ++p_;
      ws();
      if (*p_ == '}') { ++p_; return v; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (*p_ != ':') fail("expected :");
        ++p_;
        v.obj.emplace_back(std::move(k), value());
        ws();
        if (*p_ == ',') { ++p_; continue; }
        if (*p_ == '}') { ++p_; break; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = Value::Array;
      ++p_;
      ws();
      if (*p_ == ']') { ++p_; return v; }
      while (true) {
        v.arr.push_back(value());
        ws();
        if (*p_ == ',') { ++p_; continue; }
        if (*p_ == ']') { ++p_; break; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = Value::String;
      v.str = string();
    } else if (c == 't' || c == 'f') {
      v.kind = Value::Bool;
      v.b = c == 't';
      p_ += v.b ? 4 : 5;
    } else if (c == 'n') {
      p_ += 4;
    } else {
      v.kind = Value::Number;
      char *end;
      v.num = strtod(p_, &end);
      if (end == p_) fail("bad number");
      p_ = end;
    }
    return v;
  }
};

inline Value parse(const std::string &s) { return Parser(s.data(), s.data() + s.size()).parse(); }
inline Value parse(const char *p, size_t n) { return Parser(p, p + n).parse(); }

// JSON string literal (with quotes), escaping control characters; UTF-8 passes through.
inline std::string quote(const std::string &s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += (char)c;
        }
    }
  }
  return o + "\"";
}

std::string read_file(const std::string &path);

}  // namespace json
