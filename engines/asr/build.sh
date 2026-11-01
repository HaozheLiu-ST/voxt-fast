#!/bin/bash
# Builds build/bin/asr_engine and build/bin/asr.metallib (loaded from the binary's directory).
# Every kernels/*.metal is compiled with MLX's flags (-fno-fast-math) into one metallib; the host
# code is src/*.mm and src/*.cpp. Fails on any compile error.
set -euo pipefail
cd "$(dirname "$0")"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
MIN_MACOS=15.0
B=build
mkdir -p $B/air $B/obj $B/bin

METAL_FLAGS=(-x metal -mmacosx-version-min=$MIN_MACOS -fno-fast-math -Wall -Wextra
             -Wno-c++17-extensions -Wno-c++20-extensions -Wno-unused-parameter -Wno-unused-variable
             -Wno-sign-compare -Wno-unused-function -Ikernels/third_party -Ikernels)
airs=()
pids=()
for f in kernels/*.metal; do
  out=$B/air/$(basename "$f" .metal).air
  xcrun -sdk macosx metal "${METAL_FLAGS[@]}" -c "$f" -o "$out" &
  pids+=($!)
  airs+=("$out")
done
for p in "${pids[@]}"; do wait "$p" || { echo "metal compile failed" >&2; exit 1; }; done
xcrun -sdk macosx metallib "${airs[@]}" -o $B/asr.metallib

CXX=(clang++ -std=c++20 -O3 -fobjc-arc -mmacosx-version-min=$MIN_MACOS -Wall -Wno-unused-function
     -Wno-unused-variable -Isrc -DNDEBUG)
objs=()
pids=()
for f in src/*.mm src/*.cpp; do
  n=$(basename "$f"); out=$B/obj/${n%.*}.o
  "${CXX[@]}" -c "$f" -o "$out" &
  pids+=($!)
  objs+=("$out")
done
for p in "${pids[@]}"; do wait "$p" || { echo "host compile failed" >&2; exit 1; }; done
clang++ -mmacosx-version-min=$MIN_MACOS "${objs[@]}" -framework Metal -framework Foundation \
  -framework Accelerate -o $B/asr_engine
mv $B/asr.metallib $B/bin/asr.metallib
mv $B/asr_engine $B/bin/asr_engine
echo "built $PWD/$B/bin/asr_engine"
