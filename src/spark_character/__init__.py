"""spark-character: provider-agnostic voice and character for Spark agents."""

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
from .voice_judge import (
    DistinctivenessScore,
    score_distinctiveness,
    score_distinctiveness_async,
)

__version__ = "0.2.0"

__all__ = [
    "CriticSpec",
    "CritiqueResult",
    "DistinctivenessScore",
    "GenerationResult",
    "PROBES",
    "PersonaScore",
    "PersonaSpec",
    "Probe",
    "ProbeResult",
    "ProviderSpec",
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
