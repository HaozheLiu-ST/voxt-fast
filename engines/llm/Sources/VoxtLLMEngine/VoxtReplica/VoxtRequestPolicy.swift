// Request policy for a Voxt local-LLM enhancement ("transcript cleanup") request.
//
// Every value here is transcribed from the Voxt sources (read-only reference):
//   - Core/Models/CustomLLMModelSupport.swift   CustomLLMModelBehavior, CustomLLMModelFamily,
//                                                CustomLLMModelCatalog.canonicalModelRepo /
//                                                requiresVLMFactory
//   - Core/Models/CustomLLMRequestRuntime.swift  generationParameters(for:behavior:settings:tuning:)
//                                                localThinkingAdditionalContext(...)
// Only the enhancement request with default LLMGenerationSettings (all nil, thinking
// .providerDefault) and default CustomLLMGenerationTuning (no overrides) is modeled.

import Foundation
import MLXLMCommon

nonisolated enum VoxtModelFamily: String, Sendable {
    case qwen3
    case glm4
    case mistral
    case gemma
    case other

    /// CustomLLMModelFamily.resolve(for:)
    static func resolve(for repo: String) -> VoxtModelFamily {
        let normalizedRepo = VoxtModelCatalog.canonicalModelRepo(repo).lowercased()
        if normalizedRepo.contains("qwen3") { return .qwen3 }
        if normalizedRepo.contains("glm-4") || normalizedRepo.contains("glm4") {
            return .glm4
        }
        if normalizedRepo.contains("mistral") || normalizedRepo.contains("ministral") { return .mistral }
        if normalizedRepo.contains("gemma") { return .gemma }
        return .other
    }
}

nonisolated enum VoxtModelCatalog {
    static let defaultModelRepo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

    /// CustomLLMModelCatalog.compatibilityAliases
    private static let compatibilityAliases: [String: String] = [
        "Qwen/Qwen2-1.5B-Instruct": "mlx-community/Qwen3.5-2B-4bit",
        "Qwen/Qwen2.5-3B-Instruct": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit": "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3-0.6B-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3-1.7B-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3-4B-4bit": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen3-8B-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3.5-4B-4bit": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen3.5-0.8B-OptiQ-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/gemma-2-2b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-2-9b-it-4bit": "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/gemma-3-1b-it-qat-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-3n-E2B-it-lm-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-3n-E4B-it-lm-4bit": "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/Qwen3-30B-A3B-4bit": "mlx-community/Qwen3.6-27B-4bit",
        "Qwen/Qwen3-8B-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "Qwen/Qwen2.5-7B-Instruct": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3.5-2B-MLX-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3.5-0.8B-4bit-OptiQ": "mlx-community/Qwen3.5-2B-4bit",
    ]

    /// CustomLLMModelCatalog.canonicalModelRepo(_:)
    static func canonicalModelRepo(_ repo: String) -> String {
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultModelRepo }
        return compatibilityAliases[trimmed] ?? trimmed
    }

    /// CustomLLMModelCatalog.requiresVLMFactory(repo:)
    static func requiresVLMFactory(repo: String) -> Bool {
        switch canonicalModelRepo(repo) {
        case "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
             "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
             "mlx-community/gemma-4-e2b-it-4bit",
             "mlx-community/gemma-4-e4b-it-4bit",
             "mlx-community/gemma-4-12B-it-OptiQ-4bit":
            return true
        default:
            return false
        }
    }
}

nonisolated enum VoxtRequestPolicy {
    /// CustomLLMModelBehavior.thinkingOffAdditionalContext. With default settings
    /// (thinking mode .providerDefault) localThinkingAdditionalContext returns this value.
    static var thinkingOffAdditionalContext: [String: any Sendable] {
        [
            "enable_thinking": false,
            "reasoning_effort": "low",
        ]
    }

    /// CustomLLMRequestRuntime.generationParameters for an enhancement request with
    /// default settings and tuning, and the output budget supplied by the caller.
    ///
    /// `instructions` is the system prompt (request.instructions) and `prompt` is the user
    /// message (request.prompt). Character counts are Swift `String.count` (grapheme
    /// clusters), exactly as in Voxt.
    static func generationParameters(
        instructions: String,
        prompt: String,
        maxTokens: Int,
        family: VoxtModelFamily
    ) -> GenerateParameters {
        let totalPromptCharacters = instructions.count + prompt.count
        let budget = max(1, maxTokens)

        let prefillStepSize: Int
        switch totalPromptCharacters {
        case ..<1000:
            prefillStepSize = 256
        case ..<3000:
            prefillStepSize = 512
        default:
            prefillStepSize = 768
        }

        // settings.repetitionPenalty is nil by default.
        let repetitionPenalty: Float? = family == .qwen3 ? 1.05 : nil

        return GenerateParameters(
            maxTokens: budget,
            temperature: 0,
            topP: 1.0,
            topK: 0,
            minP: 0,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: 32,
            prefill: .init(stepSize: prefillStepSize, chunking: .remainder)
        )
    }
}
