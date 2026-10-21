"""Model-name -> synthcity-plugin-name mapping (Phase 4).

This mapping is **authoritative** -- ``config/model_capabilities.json``'s
``class_path`` fields mostly point at an external ``table-synthesizers``
package that does not exist in this repo (verified: ``PATECTGAN`` claims
``"table-synthesizers"`` as its whole class_path with no module). Do not
derive plugin names from that file; this module is the single source of
truth for "what can `SynthcityRunner` actually run."

Confirmed live against the installed synthcity (0.2.12, editable checkout
at ``~/Projects/synthcity``) on 2026-07-09:
- All 9 mapped plugins load and one (``arf``) was smoke-tested end to end
  (fit + generate) on a real 200-row dataset.
- ``bayesian_network`` is present in synthcity's source but its module
  fails to import in *this* environment: ``cannot import name '_lazywhere'
  from 'scipy._lib._util'`` (a scipy/pgmpy version incompatibility, not a
  Synthony bug). The mapping stays -- BayesianNetwork is a legitimate
  design target -- but training it will fail at runtime here, which the
  orchestrator's existing "training failure = candidate failure, try next
  ranked candidate" policy already handles gracefully.
- synthcity has no ``smote`` plugin at all (nor does CART/DPCART/TabSyn/
  AutoDiff map to anything) -- these need a different runner (e.g. an
  imblearn-based mini-runner for SMOTE) that doesn't exist yet, so they're
  not executable in v1.
"""

from __future__ import annotations

SYNTHCITY_PLUGIN_MAP: dict[str, str] = {
    "CTGAN": "ctgan",
    "TVAE": "tvae",
    "TabDDPM": "ddpm",
    "ARF": "arf",
    "BayesianNetwork": "bayesian_network",
    "NFlow": "nflow",
    "GReaT": "great",
    "AIM": "aim",
    "Identity": "dummy_sampler",
    "PATECTGAN": "pategan",
}

# Models in config/model_capabilities.json with no runner mapping in v1.
# Kept explicit (rather than "anything not in SYNTHCITY_PLUGIN_MAP") so the
# completeness test below fails loudly if a new model is added to the
# capabilities registry and nobody decided which bucket it belongs in.
NOT_EXECUTABLE_MODELS = frozenset({"CART", "DPCART", "TabSyn", "AutoDiff", "SMOTE"})


def executable_models() -> frozenset[str]:
    """Model names a runner can currently train, per this mapping.

    Note this describes the *design* mapping, not live plugin-import health
    in the current process -- ``BayesianNetwork`` is included even though
    it's known to fail to import in this environment (see module docstring).
    A bad mapping surfaces as a training failure, which the orchestrator
    already treats as "try the next ranked candidate," not a crash.
    """
    return frozenset(SYNTHCITY_PLUGIN_MAP)


def get_plugin_name(model_name: str) -> str | None:
    return SYNTHCITY_PLUGIN_MAP.get(model_name)
