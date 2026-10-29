"""Agentic orchestration mode.

Phase 2: monitoring.py — StepRecorder, the single instrumentation choke
point every agent step is recorded through, and the JSON log formatter for
logs/runs.jsonl.

Phase 3 (current): context.py (RunConfig/RunState/AgentContext), base.py
(BaseAgent), profiler.py (ProfilerAgent), tools.py (the tools exposed to
the reasoning loop), recommender.py (RecommenderAgent's multi-turn loop),
errors.py (RunCancelledError, is_transient), orchestrator.py (Orchestrator,
RunStatus), manager.py (RunManager) — advise mode:
PENDING -> PROFILING -> REASONING -> COMPLETED/FAILED/CANCELLED.
See docs/agents.md.

Phase 4+ (planned): runners/ package, synthesizer.py, benchmark.py — full
mode (train -> generate -> benchmark -> iterate).
"""
