"""Registry-vs-capabilities-JSON completeness (Phase 4).

Every model in ``config/model_capabilities.json`` -- the same registry
``list_candidate_models``/``get_model_details`` surface to the LLM -- must
be accounted for in ``runners/registry.py``: either it has a synthcity
plugin mapping, or it's explicitly listed in ``NOT_EXECUTABLE_MODELS``. If
someone adds a 16th model to the capabilities registry and forgets to
decide which bucket it belongs in, this test fails loudly instead of the
gap surfacing silently as "the LLM thinks it's executable but training
just always fails."
"""

from __future__ import annotations

from synthony.recommender.engine import ModelRecommendationEngine
from synthony.runners.registry import (
    NOT_EXECUTABLE_MODELS,
    SYNTHCITY_PLUGIN_MAP,
    executable_models,
    get_plugin_name,
)


def _capability_model_names() -> set[str]:
    engine = ModelRecommendationEngine()
    return set(engine.model_capabilities.get("models", {}))


class TestRegistryCompleteness:
    def test_every_capability_model_is_mapped_or_explicitly_excluded(self):
        capability_models = _capability_model_names()
        accounted_for = set(SYNTHCITY_PLUGIN_MAP) | NOT_EXECUTABLE_MODELS
        missing = capability_models - accounted_for
        assert not missing, (
            f"Model(s) {sorted(missing)} exist in model_capabilities.json but aren't in "
            "either SYNTHCITY_PLUGIN_MAP or NOT_EXECUTABLE_MODELS -- decide which bucket "
            "they belong in."
        )

    def test_no_overlap_between_executable_and_not_executable(self):
        overlap = set(SYNTHCITY_PLUGIN_MAP) & NOT_EXECUTABLE_MODELS
        assert not overlap, f"Model(s) {sorted(overlap)} are in both buckets."

    def test_mapped_models_are_not_phantom_entries(self):
        """Every mapping key should correspond to a real capabilities-JSON
        model, not a typo that happens to never get exercised."""
        capability_models = _capability_model_names()
        phantom = set(SYNTHCITY_PLUGIN_MAP) - capability_models
        assert not phantom, f"Mapped model(s) {sorted(phantom)} don't exist in model_capabilities.json"

    def test_executable_models_matches_plugin_map_keys(self):
        assert executable_models() == frozenset(SYNTHCITY_PLUGIN_MAP)

    def test_get_plugin_name_roundtrips_and_returns_none_for_unknown(self):
        for model_name, plugin_name in SYNTHCITY_PLUGIN_MAP.items():
            assert get_plugin_name(model_name) == plugin_name
        assert get_plugin_name("NotARealModel") is None
