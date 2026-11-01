#!/bin/bash
# Builds build/bin/voxt-llm-engine and, next to it, mlx-swift_Cmlx.bundle (MLX's Metal library).
#
# vendor/ holds mlx-swift and mlx-swift-lm at the revisions Voxt uses with patches/*.patch
# applied; the first build fetches it. The other packages are resolved from Package.resolved.
# SOURCE_PACKAGES=<dir> builds offline from an existing SwiftPM checkouts directory.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

fetch() {  # fetch <dir> <url> <revision> <patch>
    local dir=$1 url=$2 rev=$3 patch=$4
    [ -f "$dir/Package.swift" ] && return
    rm -rf "$dir.tmp"
    git init -q "$dir.tmp"
    git -C "$dir.tmp" fetch -q --depth 1 "$url" "$rev"
    git -C "$dir.tmp" checkout -q FETCH_HEAD
    git -C "$dir.tmp" submodule -q update --init --depth 1
    git -C "$dir.tmp" apply "$PWD/$patch"
    mv "$dir.tmp" "$dir"
}
mkdir -p vendor
fetch vendor/mlx-swift https://github.com/ml-explore/mlx-swift \
    0bb916c67f4b9e5c682cbe02a42c701c93ab5021 patches/mlx-swift.patch
fetch vendor/mlx-swift-lm https://github.com/ml-explore/mlx-swift-lm \
    c6446cf7bfb7cea76408013b614d4b2c530eaa03 patches/mlx-swift-lm.patch

offline=()
if [ -n "${SOURCE_PACKAGES:-}" ]; then
    if [ ! -d build/SourcePackages ]; then
        mkdir -p build
        cp -c -R "$SOURCE_PACKAGES" build/SourcePackages
        chmod -R u+w build/SourcePackages
    fi
    offline=(-clonedSourcePackagesDirPath build/SourcePackages -disableAutomaticPackageResolution)
fi

# xcodebuild (unlike `swift build`) compiles mlx-swift's Metal shaders into the bundle.
mkdir -p build
if ! nice -n 10 xcodebuild build \
    -scheme voxt-llm-engine \
    -configuration Release \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath build/xcode \
    -onlyUsePackageVersionsFromResolvedFile \
    -skipMacroValidation \
    -skipPackagePluginValidation \
    "${offline[@]}" > build/xcodebuild.log 2>&1; then
    tail -60 build/xcodebuild.log
    echo "build failed; full log in $PWD/build/xcodebuild.log" >&2
    exit 1
fi

products=build/xcode/Build/Products/Release
rm -rf build/bin.new
mkdir -p build/bin.new
cp "$products/voxt-llm-engine" build/bin.new/
cp -R "$products/mlx-swift_Cmlx.bundle" build/bin.new/
rm -rf build/bin
mv build/bin.new build/bin
echo "built $PWD/build/bin/voxt-llm-engine"
