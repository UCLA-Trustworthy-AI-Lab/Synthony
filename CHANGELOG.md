# Changelog

All notable changes to Synthony are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-07-01

Documentation and CI/CD accuracy pass. No public API changes.

### Fixed
- Removed stale references to the `GaussianCopula` model (dropped from the
  registry in a prior update) across tests, scripts, and docstrings;
  replaced with the current small-data model `CART`
- Fixed `"PATE-CTGAN"` naming to `"PATECTGAN"` and made model-name
  assertions in `tests/functional/test_recommendation_methods.py` derive
  from the live registry instead of a hardcoded list
- Fixed `mcp_server/test_server.py`, which used `return True/False`
  instead of `assert` — a hard failure under pytest >=8
- Corrected several `docs/` planning documents that described a
  never-built "3-package ecosystem," phantom `/sessions` REST endpoints,
  a stale Zipfian threshold (`0.05` instead of the actual `0.80`), and
  an incorrect `gpu_recommendation` field name (actual: `requires_gpu`)
- Documented the `baselines/` directory and the `config/` vs
  package-local `model_capabilities.json` sync footgun in `CLAUDE.md`
- Fixed a stale "13+ models" claim in the API description (now 15)

### Added
- `tests/regression/` (synthetic-data-only) wired into the CI pipeline
- `CHANGELOG.md`

Published at the [2nd DeLTa Workshop, ICLR 2026](https://openreview.net/forum?id=cj4SNumWqf).
