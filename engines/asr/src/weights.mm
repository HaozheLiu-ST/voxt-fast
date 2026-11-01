#include "weights.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace json {
std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace json

static gpu::DType parse_dtype(const std::string &s) {
  if (s == "F32") return gpu::DType::F32;
  if (s == "F16") return gpu::DType::F16;
  if (s == "BF16") return gpu::DType::BF16;
  if (s == "U32") return gpu::DType::U32;
  if (s == "I32") return gpu::DType::I32;
  if (s == "U8") return gpu::DType::U8;
  throw std::runtime_error("unsupported safetensors dtype " + s);
}

Weights::Weights(const std::string &model_dir) : dir(model_dir) {
  config = json::parse(json::read_file(model_dir + "/config.json"));
  std::vector<std::string> files;
  if (DIR *d = opendir(model_dir.c_str())) {
    while (dirent *e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") files.push_back(model_dir + "/" + n);
    }
    closedir(d);
  }
  if (files.empty()) throw std::runtime_error("no .safetensors in " + model_dir);
  std::sort(files.begin(), files.end());
  id<MTLDevice> dev = gpu::Context::get().device;
  for (auto &path : files) {
    // The data region of these files starts at an odd offset (8 + header length), so tensors cannot be
    // used in place: read the data region into one page-aligned shared MTLBuffer.
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    uint64_t hlen = 0;
    if (pread(fd, &hlen, 8, 0) != 8) throw std::runtime_error("short read " + path);
    std::string header(hlen, '\0');
    if (pread(fd, header.data(), hlen, 8) != (ssize_t)hlen) throw std::runtime_error("short read " + path);
    json::Value h = json::parse(header);
    size_t data0 = 8 + hlen;
    size_t data_len = (size_t)st.st_size - data0;
    id<MTLBuffer> buf = [dev newBufferWithLength:(data_len + 16383) / 16384 * 16384 options:MTLResourceStorageModeShared];
    if (!buf) throw std::runtime_error("weight buffer allocation failed " + path);
    char *dstp = (char *)[buf contents];
    size_t done = 0;
    while (done < data_len) {
      size_t n = std::min<size_t>(data_len - done, (size_t)1 << 28);
      ssize_t r = pread(fd, dstp + done, n, data0 + done);
      if (r <= 0) throw std::runtime_error("read failed " + path);
      done += (size_t)r;
    }
    close(fd);
    for (auto &[name, v] : h.obj) {
      if (name == "__metadata__") continue;
      gpu::Tensor t;
      t.buf = (__bridge id)CFRetain((__bridge CFTypeRef)buf);  // every view owns the shared file buffer
      t.dtype = parse_dtype(v["dtype"].as_str());
      for (auto &s : v["shape"].arr) t.shape.push_back(s.as_int());
      t.offset = (size_t)v["data_offsets"][0].as_int();
      tensors[name] = t;
    }
  }
}

const gpu::Tensor &Weights::get(const std::string &name) const {
  auto it = tensors.find(name);
  if (it == tensors.end()) throw std::runtime_error("missing weight " + name);
  return it->second;
}
