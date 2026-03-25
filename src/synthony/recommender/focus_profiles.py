"""Focus-based scale factor profiles for the recommendation engine.

Each focus (privacy, fidelity, latency) defines scale factors that multiply
the base capability weights in _score_models(). This allows the engine to
prioritize different aspects of model selection based on user intent.

Scale factors default to 1.0 (no behavior change). After Bayesian optimization,
they are updated to values that maximize Top-1 accuracy against ground truth.
"""

from typing import Dict

CAPABILITY_NAMES = [
    "skew_handling",
    "cardinality_handling",
    "zipfian_handling",
    "small_data",
    "correlation_handling",
    "privacy_dp",
]

# Default: all 1.0 (no behavior change from existing engine)
FOCUS_REGISTRY: Dict[str, Dict[str, float]] = {
    "privacy": {cap: 1.0 for cap in CAPABILITY_NAMES},
    "fidelity": {cap: 1.0 for cap in CAPABILITY_NAMES},
    "latency": {cap: 1.0 for cap in CAPABILITY_NAMES},
}


def get_scale_factors(focus: str) -> Dict[str, float]:
    """Look up scale factors for a named focus. Returns a copy.

    Args:
        focus: Focus name (e.g. "privacy", "fidelity", "latency")

    Returns:
        Dict mapping capability name to scale factor (float).

    Raises:
        KeyError: If focus name is not registered.
    """
    if focus not in FOCUS_REGISTRY:
        raise KeyError(
            f"Unknown focus '{focus}'. Available: {list(FOCUS_REGISTRY.keys())}"
        )
    return dict(FOCUS_REGISTRY[focus])


def register_focus(name: str, scale_factors: Dict[str, float]) -> None:
    """Register or update a focus profile.

    Args:
        name: Focus name (e.g. "privacy").
        scale_factors: Dict mapping capability names to scale factors.
            Missing capabilities default to 1.0.
    """
    full = {cap: scale_factors.get(cap, 1.0) for cap in CAPABILITY_NAMES}
    FOCUS_REGISTRY[name] = full
