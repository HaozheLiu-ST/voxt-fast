// swift-tools-version: 6.2
//
// Voxt's local LLM cleanup request (mlx-community/Qwen3.5-4B-OptiQ-4bit) as a JSON-lines
// service, with a prefix cache and exact speculative decoding.
//
// The packages are pinned to the versions Voxt resolves (Package.resolved), so the model runs
// the same code as in Voxt apart from the changes in patches/:
//   mlx-swift-lm        revision c6446cf7bfb7cea76408013b614d4b2c530eaa03  (vendor/, patched)
//   mlx-swift           0.31.6, revision 0bb916c67f4b9e5c682cbe02a42c701c93ab5021  (vendor/, patched)
//   swift-transformers  1.3.4, swift-jinja 2.5.1, swift-huggingface 0.10.2,
//   swift-collections 1.6.0, yyjson 0.12.0, swift-numerics 1.1.1
// scripts/build.sh fetches vendor/ and applies the patches.

import PackageDescription

let package = Package(
    name: "voxt-llm-engine",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "voxt-llm-engine", targets: ["VoxtLLMEngine"])
    ],
    dependencies: [
        .package(path: "vendor/mlx-swift-lm"),
        .package(path: "vendor/mlx-swift"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.5.1"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.10.2"),
        .package(url: "https://github.com/apple/swift-collections.git", exact: "1.6.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
        .package(url: "https://github.com/apple/swift-numerics", exact: "1.1.1"),
    ],
    targets: [
        .executableTarget(
            name: "VoxtLLMEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Voxt's model loader imports MLXVLM for its vision-language branch.
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "yyjson", package: "yyjson"),
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Sources/VoxtLLMEngine",
            swiftSettings: [
                // The Voxt app target's settings, so the files copied from Voxt keep their
                // actor isolation.
                .swiftLanguageMode(.v5),
                .defaultIsolation(MainActor.self),
                .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
                .enableUpcomingFeature("InferIsolatedConformances"),
                .enableUpcomingFeature("InferSendableFromCaptures"),
                .enableUpcomingFeature("GlobalActorIsolatedTypesUsability"),
                .enableUpcomingFeature("DisableOutwardActorInference"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ]
        )
    ]
)
