"""spark-character: provider-agnostic voice and character for Spark agents."""

from .audit_miner import (
    AuditFailure,
    AuditFindings,
    AuditMiner,
)
from .chip_context import (
    attach_chip_context,
    chip_context_for,
    known_chip_keys,
)
from .chip_loader import (
    PersonalityChip,
    load_chip,
    load_chip_by_id,
    persona_from_chip,
    render_chip_to_system_prompt,
)
from .codex_provider import (
    CodexSpec,
    call_codex,
    codex_available,
)
from .critic import (
    CriticSpec,
    CritiqueResult,
    critique,
    critique_async,
    load_critic,
)
from .persona import (
    PersonaSpec,
    load_persona,
    load_persona_from_path,
)
from .pipeline import (
    GenerationResult,
    generate,
    generate_async,
    generate_with_critique,
    generate_with_critique_async,
)
from .provider import (
    ProviderSpec,
    call_provider,
    call_provider_async,
)
from .deeper_probes import (
    T6_EMOTIONAL_ATTUNEMENT_PROBES,
    T7_MEMORY_COHERENCE_PROBES,
    T8_INITIATIVE_PROBES,
    DeepProbe,
    DeepProbeResult,
    run_deep_probe,
)
from .probes import (
    PROBES,
    Probe,
    ProbeResult,
    run_probe,
)
from .scoring import (
    PersonaScore,
    score_persona,
)
from .stability import (
    STABILITY_SCENARIOS,
    StabilityResult,
    StabilityScenario,
    run_stability_scenario,
)
from .voice_judge import (
    DistinctivenessScore,
    score_distinctiveness,
    score_distinctiveness_async,
)

__version__ = "0.3.0"

__all__ = [
    "AuditFailure",
    "AuditFindings",
    "AuditMiner",
    "CodexSpec",
    "CriticSpec",
    "CritiqueResult",
    "call_codex",
    "codex_available",
    "DistinctivenessScore",
    "GenerationResult",
    "PROBES",
    "PersonaScore",
    "PersonaSpec",
    "PersonalityChip",
    "Probe",
    "ProbeResult",
    "ProviderSpec",
    "attach_chip_context",
    "chip_context_for",
    "known_chip_keys",
    "load_chip",
    "load_chip_by_id",
    "persona_from_chip",
    "render_chip_to_system_prompt",
    "STABILITY_SCENARIOS",
    "T6_EMOTIONAL_ATTUNEMENT_PROBES",
    "T7_MEMORY_COHERENCE_PROBES",
    "T8_INITIATIVE_PROBES",
    "DeepProbe",
    "DeepProbeResult",
    "run_deep_probe",
    "StabilityResult",
    "StabilityScenario",
    "run_stability_scenario",
    "call_provider",
    "call_provider_async",
    "critique",
    "critique_async",
    "generate",
    "generate_async",
    "generate_with_critique",
    "generate_with_critique_async",
    "load_critic",
    "load_persona",
    "load_persona_from_path",
    "run_probe",
    "score_distinctiveness",
    "score_distinctiveness_async",
    "score_persona",
    "__version__",
]
