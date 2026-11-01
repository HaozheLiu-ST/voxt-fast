// Safetensors weights, memory-mapped and wrapped zero-copy in one MTLBuffer per file.
#pragma once
#include <map>
#include <string>

#include "gpu.h"
#include "json.h"

struct Weights {
  // Loads every *.safetensors in `model_dir`; also parses config.json into `config`.
  explicit Weights(const std::string &model_dir);
  bool has(const std::string &name) const { return tensors.count(name) != 0; }
  const gpu::Tensor &get(const std::string &name) const;  // throws if missing
  std::map<std::string, gpu::Tensor> tensors;
  json::Value config;
  std::string dir;
};
