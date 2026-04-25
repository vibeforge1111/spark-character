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
    load_overlay,
    load_persona,
    load_persona_from_path,
    load_surface_overlay,
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
    T9_AESTHETIC_FINGERPRINT_PROBES,
    T13_HUMANE_DEPTH_PROBES,
    T14_MEMORABILITY_PROBES,
    DeepProbe,
    DeepProbeResult,
    run_deep_probe,
)
from .memory_grounded import (
    UserInstruction,
    UserStateObservation,
    build_t7_probes_from_state,
    latest_user_instructions,
    latest_user_states,
    memory_grounded_summary,
    state_distribution,
)
from .registry import (
    find_chip_lab_path,
    promote_evolved_chip_to_chip_lab,
    promote_evolved_persona_to_chip_lab,
)
from .trait_mutator import (
    EMOTIONAL_PROFILE_FIELDS,
    EMOTIONAL_RANGE_KEYS,
    MAX_DELTA_PER_TRAIT,
    TRAIT_FIELDS,
    TraitMutationResult,
    chip_to_yaml_dict,
    mutate_trait_values,
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
from .search_adapter import (
    SearchResult,
    attach_search_context,
    detect_needs_live_data,
    extract_search_query,
    search_results_for,
)
from .stability import (
    STABILITY_SCENARIOS,
    T11_SUSTAINED_ATTACK_SCENARIOS,
    StabilityResult,
    StabilityScenario,
    run_stability_scenario,
)
from .output_sanitizer import (
    EM_DASH_FAMILY,
    replace_em_dashes,
    sanitize_voice_output,
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
    "EM_DASH_FAMILY",
    "replace_em_dashes",
    "sanitize_voice_output",
    "GenerationResult",
    "PROBES",
    "PersonaScore",
    "PersonaSpec",
    "PersonalityChip",
    "Probe",
    "ProbeResult",
    "ProviderSpec",
    "SearchResult",
    "attach_search_context",
    "detect_needs_live_data",
    "extract_search_query",
    "search_results_for",
    "attach_chip_context",
    "chip_context_for",
    "find_chip_lab_path",
    "known_chip_keys",
    "promote_evolved_chip_to_chip_lab",
    "promote_evolved_persona_to_chip_lab",
    "load_chip",
    "load_chip_by_id",
    "load_overlay",
    "load_persona",
    "load_persona_from_path",
    "load_surface_overlay",
    "persona_from_chip",
    "render_chip_to_system_prompt",
    "EMOTIONAL_PROFILE_FIELDS",
    "EMOTIONAL_RANGE_KEYS",
    "MAX_DELTA_PER_TRAIT",
    "STABILITY_SCENARIOS",
    "T11_SUSTAINED_ATTACK_SCENARIOS",
    "T6_EMOTIONAL_ATTUNEMENT_PROBES",
    "T7_MEMORY_COHERENCE_PROBES",
    "T8_INITIATIVE_PROBES",
    "T9_AESTHETIC_FINGERPRINT_PROBES",
    "T13_HUMANE_DEPTH_PROBES",
    "T14_MEMORABILITY_PROBES",
    "TRAIT_FIELDS",
    "TraitMutationResult",
    "chip_to_yaml_dict",
    "mutate_trait_values",
    "DeepProbe",
    "DeepProbeResult",
    "UserInstruction",
    "UserStateObservation",
    "build_t7_probes_from_state",
    "latest_user_instructions",
    "latest_user_states",
    "memory_grounded_summary",
    "run_deep_probe",
    "state_distribution",
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
