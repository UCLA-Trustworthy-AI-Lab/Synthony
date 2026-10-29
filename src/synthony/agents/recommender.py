"""RecommenderAgent -- the REASONING stage of an advise-mode run.

A direct multi-turn tool-use loop against the Phase 1 ``LLMClient``. No
pluggable reasoning-engine protocol/registry: this agent owns the loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from synthony.agents.base import BaseAgent
from synthony.agents.errors import RunCancelledError, is_transient
from synthony.agents.tools import (
    SUBMIT_RECOMMENDATION_SPEC,
    build_recommender_tools,
    validate_submission,
)
from synthony.llm import LLMMessage, ToolResult
from synthony.runners.registry import executable_models

_MAX_SCHEMA_REPAIRS = 2
_MAX_TRANSIENT_RETRIES = 2

_REASONING_PROTOCOL = """\
## Agentic Reasoning Protocol

You are recommending a synthetic-data generation model for a dataset via \
tool calls. Available tools: `get_profile_overview`, `get_column_stats`, \
`list_candidate_models`, `get_model_details`, `get_rule_based_scores`, \
`get_prior_attempts`, and the terminal `submit_recommendation`.

Rules:
1. Call `get_profile_overview` before anything else.
2. Call `get_rule_based_scores` at least once before submitting -- treat it \
as a deterministic anchor, not something to blindly copy.
3. When ready, call `submit_recommendation` with 1-3 ranked candidates \
(`model_name`, `confidence` 0-1, `hyperparameter_hints`, `rationale`) plus \
overall `reasoning`. This ends the conversation.
4. You have a limited number of turns and tokens -- do not call the same \
tool repeatedly with identical arguments.
"""


class SchemaRepairExhaustedError(Exception):
    """Raised internally when submit_recommendation keeps failing validation."""


class RecommenderAgent(BaseAgent):
    name = "recommender"

    async def run(self) -> dict[str, Any]:
        state = self.context.state
        cfg = state.config

        if self.context.llm_client is None:
            # No LLM configured -- don't attempt reasoning, go straight to the anchor.
            return await self._fallback_to_rule_based(
                "Rule-based recommendation "
                "(no LLM configured: llm_provider=none or no credentials)."
            )

        tools = build_recommender_tools()
        self._handlers = {t.spec.name: t.handler for t in tools}
        self._tool_specs = [t.spec for t in tools] + [SUBMIT_RECOMMENDATION_SPEC]
        self._repairs_used = 0

        system = self._build_system_prompt()
        state.conversation = [LLMMessage(role="user", content=self._build_goal_prompt())]

        turn = 0
        try:
            async with asyncio.timeout(cfg.step_timeout_s):
                while turn < cfg.max_reasoning_turns and state.tokens_used < cfg.max_run_tokens:
                    if state.cancel_event.is_set():
                        raise RunCancelledError()
                    turn += 1
                    result = await self._run_turn(system, turn)
                    if result is not None:
                        return result
        except TimeoutError:
            pass
        except SchemaRepairExhaustedError:
            pass

        forced = await self._force_submit(system, turn + 1)
        if forced is not None:
            return forced
        return await self._fallback_to_rule_based()

    # -- one regular turn, with transient-error retry -----------------------

    async def _run_turn(self, system: str, turn: int) -> dict[str, Any] | None:
        state = self.context.state
        step_name = f"reason_turn_{turn}"
        attempt = 0
        while True:
            try:
                async with state.recorder.step(state.run_id, self.name, step_name) as handle:
                    response = await self.context.llm_client.complete(
                        state.conversation,
                        system=system,
                        tools=self._tool_specs,
                        max_tokens=4096,
                    )
                    handle.add_usage(response.usage)
                    state.tokens_used += response.usage.total_tokens
                    state.conversation.append(response.to_assistant_message())
                    handle.set_output(
                        {
                            "stop_reason": response.stop_reason,
                            "tool_calls": [tc.name for tc in response.tool_calls],
                        }
                    )
                break
            except Exception as exc:
                if is_transient(exc) and attempt < _MAX_TRANSIENT_RETRIES:
                    attempt += 1
                    step_name = f"reason_turn_{turn}_retry_{attempt}"
                    await asyncio.sleep(2**attempt)
                    continue
                raise
        return await self._handle_response(response)

    async def _handle_response(self, response) -> dict[str, Any] | None:
        state = self.context.state
        submit_call = next(
            (tc for tc in response.tool_calls if tc.name == "submit_recommendation"), None
        )

        if submit_call is not None:
            try:
                payload = validate_submission(submit_call.arguments)
            except ValidationError as exc:
                self._repairs_used += 1
                if self._repairs_used > _MAX_SCHEMA_REPAIRS:
                    raise SchemaRepairExhaustedError() from exc
                state.conversation.append(
                    LLMMessage(
                        role="tool",
                        tool_results=[
                            ToolResult(
                                tool_call_id=submit_call.id,
                                content=f"Invalid submission, please fix and resubmit: {exc}",
                                is_error=True,
                            )
                        ],
                    )
                )
                return None

            if state.config.mode == "full":
                top_model = payload.ranked_candidates[0].model_name
                if top_model not in executable_models():
                    self._repairs_used += 1
                    if self._repairs_used > _MAX_SCHEMA_REPAIRS:
                        raise SchemaRepairExhaustedError()
                    state.conversation.append(
                        LLMMessage(
                            role="tool",
                            tool_results=[
                                ToolResult(
                                    tool_call_id=submit_call.id,
                                    content=(
                                        f"'{top_model}' cannot be executed by the runner in "
                                        f"mode='full'. Choose a top candidate from: "
                                        f"{sorted(executable_models())}"
                                    ),
                                    is_error=True,
                                )
                            ],
                        )
                    )
                    return None

            result = self._to_result(payload, degraded=False)
            state.final_recommendation = result
            return result

        if response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                handler = self._handlers.get(tc.name)
                if handler is None:
                    tool_results.append(
                        ToolResult(
                            tool_call_id=tc.id, content=f"Unknown tool '{tc.name}'", is_error=True
                        )
                    )
                    continue
                try:
                    output = await handler(self.context, tc.arguments)
                except Exception as exc:  # tools shouldn't raise, but never let one crash the loop
                    tool_results.append(
                        ToolResult(tool_call_id=tc.id, content=f"Tool raised: {exc}", is_error=True)
                    )
                    continue
                is_error = bool(output.pop("is_error", False))
                tool_results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=json.dumps(output, default=str),
                        is_error=is_error,
                    )
                )
            state.conversation.append(LLMMessage(role="tool", tool_results=tool_results))
            return None

        state.conversation.append(
            LLMMessage(
                role="user", content="Please call a tool, or submit_recommendation when ready."
            )
        )
        return None

    # -- forced submit + graceful fallback -----------------------------------

    async def _force_submit(self, system: str, turn: int) -> dict[str, Any] | None:
        state = self.context.state
        step_name = f"reason_turn_{turn}_forced"
        try:
            async with state.recorder.step(state.run_id, self.name, step_name) as handle:
                response = await self.context.llm_client.complete(
                    state.conversation,
                    system=system,
                    tools=self._tool_specs,
                    tool_choice="submit_recommendation",
                    max_tokens=4096,
                )
                handle.add_usage(response.usage)
                state.tokens_used += response.usage.total_tokens
                handle.set_output({"stop_reason": response.stop_reason})
        except Exception:
            return None  # single attempt -- any failure here falls through to rule-based fallback

        submit_call = next(
            (tc for tc in response.tool_calls if tc.name == "submit_recommendation"), None
        )
        if submit_call is None:
            return None
        try:
            payload = validate_submission(submit_call.arguments)
        except ValidationError:
            return None

        result = self._to_result(payload, degraded=False)
        state.final_recommendation = result
        return result

    async def _fallback_to_rule_based(
        self,
        reasoning: str = "Rule-based fallback (LLM reasoning did not produce a valid submission).",
    ) -> dict[str, Any]:
        state = self.context.state
        async with state.recorder.step(state.run_id, self.name, "fallback_rule_based") as handle:
            result = state.rule_based_result
            if result is None:
                result = self.context.engine.recommend(
                    state.dataset_profile,
                    state.column_analysis,
                    constraints=state.config.constraints,
                    method="rule_based",
                )
                state.rule_based_result = result
            candidates = [result.recommended_model, *result.alternative_models[:2]]
            payload = {
                "ranked_candidates": [
                    {
                        "model_name": c.model_name,
                        "confidence": c.confidence_score,
                        "hyperparameter_hints": {},
                        "rationale": "; ".join(c.reasoning) if c.reasoning else "",
                    }
                    for c in candidates
                ],
                "reasoning": reasoning,
                "degraded": True,
            }
            handle.set_output(
                {"degraded": True, "recommended_model": result.recommended_model.model_name}
            )
        state.final_recommendation = payload
        return payload

    @staticmethod
    def _to_result(payload, *, degraded: bool) -> dict[str, Any]:
        return {
            "ranked_candidates": [c.model_dump() for c in payload.ranked_candidates],
            "reasoning": payload.reasoning,
            "degraded": degraded,
        }

    # -- prompt construction --------------------------------------------------

    def _build_system_prompt(self) -> str:
        base = self.context.engine.system_prompt or ""
        return f"{base}\n\n{_REASONING_PROTOCOL}"

    def _build_goal_prompt(self) -> str:
        cfg = self.context.state.config
        lines = [
            f"Recommend a synthetic-data generation model for dataset '{cfg.dataset_id}'.",
            f"Constraints: {cfg.constraints.model_dump()}",
        ]
        if cfg.mode == "full":
            lines.append(
                "This run will actually train and generate your top-ranked model, so it "
                f"must be executable: {sorted(executable_models())}. Alternatives beyond "
                "the top candidate may be anything, for use if training fails."
            )
        if self.context.state.attempts:
            lines.append(
                f"This is iteration {self.context.state.iteration} -- call `get_prior_attempts` "
                "to see what was already tried and why it fell short before recommending again."
            )
        lines.append("Begin by calling get_profile_overview.")
        return "\n".join(lines)
