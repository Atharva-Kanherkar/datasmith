from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from asi.prompts import challenger_prompt, judge_prompt, solver_prompt
from asi.types import Example, JudgeVerdict, Model, RejectedExample, RunResult, SolverAttempt


@dataclass(slots=True)
class AcceptancePolicy:
    min_gap: float = 0.2
    max_weak_score: float = 0.5
    min_strong_score: float = 0.65
    min_quality: str = "medium"

    def accepts(self, verdict: JudgeVerdict) -> bool:
        if not verdict.accepted():
            return False
        if verdict.gap is None or verdict.gap < self.min_gap:
            return False
        if verdict.weak_score is None or verdict.weak_score > self.max_weak_score:
            return False
        if verdict.strong_score is None or verdict.strong_score < self.min_strong_score:
            return False
        return _quality_rank(verdict.quality) >= _quality_rank(self.min_quality)


@dataclass(slots=True)
class AgenticSelfInstruct:
    challenger: Model
    weak_solver: Model
    strong_solver: Model
    judge: Model
    policy: AcceptancePolicy = field(default_factory=AcceptancePolicy)
    weak_rollouts: int = 3
    strong_rollouts: int = 3
    max_attempts_per_example: int = 8
    seed_batch_size: int = 3
    random_seed: int = 7

    def run(self, seeds: list[Example], *, target_count: int) -> RunResult:
        if not seeds:
            raise ValueError("at least one seed example is required")
        if target_count < 1:
            raise ValueError("target_count must be at least 1")
        rng = random.Random(self.random_seed)
        result = RunResult()
        max_attempts = max(target_count * self.max_attempts_per_example, target_count)

        for attempt_index in range(1, max_attempts + 1):
            if len(result.accepted) >= target_count:
                break
            result.attempts = attempt_index
            seed_batch = _sample(seeds, self.seed_batch_size, rng)
            candidate, candidate_error = self._create_candidate(seed_batch, result.feedback, target_count)
            if candidate_error:
                result.rejected.append(
                    RejectedExample(
                        candidate=Example(input={"prompt_error": True}),
                        reason_code="candidate_parse_error",
                        reason=candidate_error,
                    )
                )
                continue

            weak_attempts, weak_error = self._solve(
                "weak", self.weak_solver, candidate, self.weak_rollouts
            )
            if weak_error:
                result.rejected.append(
                    RejectedExample(
                        candidate=candidate,
                        reason_code="weak_solver_error",
                        reason=weak_error,
                        weak_attempts=weak_attempts,
                    )
                )
                continue

            strong_attempts, strong_error = self._solve(
                "strong", self.strong_solver, candidate, self.strong_rollouts
            )
            if strong_error:
                result.rejected.append(
                    RejectedExample(
                        candidate=candidate,
                        reason_code="strong_solver_error",
                        reason=strong_error,
                        weak_attempts=weak_attempts,
                        strong_attempts=strong_attempts,
                    )
                )
                continue

            verdict, judge_error = self._judge(candidate, weak_attempts, strong_attempts)
            if judge_error:
                result.rejected.append(
                    RejectedExample(
                        candidate=candidate,
                        reason_code="judge_parse_error",
                        reason=judge_error,
                        weak_attempts=weak_attempts,
                        strong_attempts=strong_attempts,
                    )
                )
                continue

            if self.policy.accepts(verdict):
                candidate.metadata = {
                    **candidate.metadata,
                    "generator": "agentic-self-instruct",
                    "judge": verdict.to_dict(),
                    "weak_attempts": [attempt.to_dict() for attempt in weak_attempts],
                    "strong_attempts": [attempt.to_dict() for attempt in strong_attempts],
                }
                result.accepted.append(candidate)
            else:
                reason = verdict.reason or "judge rejected candidate"
                result.feedback.append(verdict.feedback or reason)
                result.rejected.append(
                    RejectedExample(
                        candidate=candidate,
                        reason_code="quality_rejected",
                        reason=reason,
                        weak_attempts=weak_attempts,
                        strong_attempts=strong_attempts,
                        judge=verdict,
                    )
                )

        return result

    def _create_candidate(
        self, seeds: list[Example], feedback: list[str], target_count: int
    ) -> tuple[Example, str | None]:
        prompt = challenger_prompt(seeds, feedback, target_count)
        raw = self.challenger.complete(prompt, role="challenger", metadata={"target_count": target_count})
        try:
            parsed = json.loads(_strip_fences(raw))
            if not isinstance(parsed, dict):
                return Example(input={}), "challenger returned non-object JSON"
            return Example.from_dict(parsed), None
        except Exception as exc:  # noqa: BLE001 - convert arbitrary model failures into artifacts
            return Example(input={}), f"parse challenger response: {exc}"

    def _solve(
        self, role: str, model: Model, candidate: Example, rollouts: int
    ) -> tuple[list[SolverAttempt], str | None]:
        attempts: list[SolverAttempt] = []
        for index in range(1, max(1, rollouts) + 1):
            prompt = solver_prompt(role, candidate)
            try:
                output = model.complete(
                    prompt,
                    role=f"{role}_solver",
                    metadata={"attempt": index},
                )
            except Exception as exc:  # noqa: BLE001
                return attempts, f"{role} solver attempt {index}: {exc}"
            attempts.append(SolverAttempt(role=role, output=output.strip(), attempt=index))
        return attempts, None

    def _judge(
        self,
        candidate: Example,
        weak_attempts: list[SolverAttempt],
        strong_attempts: list[SolverAttempt],
    ) -> tuple[JudgeVerdict, str | None]:
        prompt = judge_prompt(candidate, weak_attempts, strong_attempts)
        raw = self.judge.complete(prompt, role="judge", metadata={})
        try:
            parsed = json.loads(_strip_fences(raw))
            if not isinstance(parsed, dict):
                return JudgeVerdict(verdict="reject"), "judge returned non-object JSON"
            return _verdict_from_dict(parsed), None
        except Exception as exc:  # noqa: BLE001
            return JudgeVerdict(verdict="reject"), f"parse judge response: {exc}"


def _verdict_from_dict(data: dict[str, Any]) -> JudgeVerdict:
    verdict = str(data.get("verdict") or "reject")
    if verdict not in {"accept", "reject"}:
        raise ValueError("judge verdict must be accept or reject")
    return JudgeVerdict(
        verdict=verdict,
        weak_score=_optional_float(data.get("weak_score")),
        strong_score=_optional_float(data.get("strong_score")),
        gap=_optional_float(data.get("gap")),
        quality=str(data["quality"]) if data.get("quality") else None,
        reason=str(data["reason"]) if data.get("reason") else None,
        feedback=str(data["feedback"]) if data.get("feedback") else None,
        tags=[str(item) for item in data.get("tags", []) if isinstance(item, str)],
        raw=dict(data),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise ValueError("scores must be between 0 and 1")
    return parsed


def _quality_rank(value: str | None) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value or "medium", 1)


def _sample(seeds: list[Example], size: int, rng: random.Random) -> list[Example]:
    if len(seeds) <= size:
        return list(seeds)
    return rng.sample(seeds, size)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json")
    elif text.startswith("```"):
        text = text.removeprefix("```")
    if text.endswith("```"):
        text = text.removesuffix("```")
    return text.strip()
