from enum import Enum, StrEnum, auto


class ClaudeResponseType(StrEnum):
    ASSISTANT = auto()
    SYSTEM = auto()


class ClaudeResponseMessageType(StrEnum):
    TEXT = auto()


class InferenceEngineType(Enum):
    """Enum to contain possible values for inference engine types"""

    OLLAMA = auto()

    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))

    def __str__(self):
        return self.name


class InferenceModel(StrEnum):
    """Default model identifiers"""

    OLLAMA_RISK_MODEL = "granite4.1:8b"
    # Granite Guardian 4.1 (Q4_K_M quantization via community mirror).
    # Empirical audit on 2026-05-15 showed Guardian 3.3 with the default
    # policy caught 0 of 6 hallucination findings on the
    # eu-ai-act-classification fixture and produced false positives on
    # legitimate regulatory bias-discussion. 4.1 is expected to support
    # the broader native-risk set including `groundedness` and
    # `answer_relevance` (now injected as mandatory risks in
    # `nexus_policy.generate_policy_manifest`).
    #
    # Tag naming history:
    # - IBM does not (as of 2026-05-15) publish an official `ibm/granite4.1
    #   -guardian:8b` tag in the ollama registry.
    # - `elishabjm/granite-guardian-4.1:8b-q4_k_m` is a Q4_K_M-quantized
    #   community mirror (5.1 GB, vs ~6.7 GB for the 3.3-8b full-precision
    #   version). Quantization trades some accuracy for size/speed; acceptable
    #   for runtime audit checks.
    # If/when IBM publishes an official tag, update this constant and remove
    # the dependency on the community mirror.
    OLLAMA_GUARDIAN_MODEL = "elishabjm/granite-guardian-4.1:8b-q4_k_m"
    CLAUDE_MODEL = "sonnet"


class SpecFileFormat(StrEnum):
    """Default spec file identifiers"""

    SKILL_FILE_MD = "SKILL.md"
    SPEC_FILE_MD = "spec.md"


class GovernanceTaxonomy(StrEnum):
    """Default taxonomy identifiers"""

    IBM_GRANITE_GUARDIAN = "ibm-granite-guardian"
    NIST_AI_RMF = "nist-ai-rmf"
    CREDO_UCF = "credo-ucf"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class CoverageLevel(Enum):
    AUTOMATED = "AUTOMATED"
    PARTIAL = "PARTIAL"
    MANUAL = "MANUAL"
