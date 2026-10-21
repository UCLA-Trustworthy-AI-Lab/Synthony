"""Base class every Phase 3 agent implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from synthony.agents.context import AgentContext


class BaseAgent(ABC):
    """One stage of a run (profiler, recommender, ...).

    Agents mutate ``self.context.state`` in place and return a JSON-safe
    summary for the orchestrator's ``result_json``/logging. Agents never
    call ``update_run_status``/``finish_run`` themselves — persisting the
    run-level status transition is the orchestrator's exclusive
    responsibility, so "status persisted before each stage starts" stays
    enforceable in one place.
    """

    name: ClassVar[str]

    def __init__(self, context: AgentContext):
        self.context = context

    @abstractmethod
    async def run(self) -> dict[str, Any]:
        """Execute this agent's work for ``context.state``."""
