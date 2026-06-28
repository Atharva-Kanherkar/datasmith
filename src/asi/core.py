from __future__ import annotations

import json
import math
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

    def __post_init__(self) -> None:
        for name in ("min_gap", "max_weak_score", "min_strong_score"):
            _validate_score(getattr(self, name), name)
        _quality_rank(self.min_quality)

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

    def __post_init__(self) -> None:
        if self.weak_rollouts < 1:
            raise ValueError("weak_rollouts must be at least 1")
        if self.strong_rollouts < 1:
            raise ValueError("strong_rollouts must be at least 1")
        if self.max_attempts_per_example < 1:
            raise ValueError("max_attempts_per_example must be at least 1")
        if self.seed_batch_size < 1:
            raise ValueError("seed_batch_size must be at least 1")

    def run(self, seeds: list[Example], *, target_count: int) -> RunResult:
        if not seeds:
            raise ValueError("at least one seed example is required")
        if target_count < 1:
            raise ValueError("target_count must be at least 1")
        rng = random.Random(self.random_seed)
        result = RunResult(target_count=target_count)
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
                    "generator": "datasmith",
                    "judge": verdict.to_dict(),
                    "weak_attempts": [attempt.to_dict() for attempt in weak_attempts],
                    "strong_attempts": [attempt.to_dict() for attempt in strong_attempts],
                }
                result.accepted.append(candidate)
            else:
                reason = verdict.reason or "judge rejected candidate"
                reason_code = "judge_rejected" if not verdict.accepted() else "policy_rejected"
                result.feedback.append(verdict.feedback or reason)
                result.rejected.append(
                    RejectedExample(
                        candidate=candidate,
                        reason_code=reason_code,
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
        try:
            raw = self.challenger.complete(
                prompt, role="challenger", metadata={"target_count": target_count}
            )
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
        for index in range(1, rollouts + 1):
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
        try:
            raw = self.judge.complete(prompt, role="judge", metadata={})
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
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("judge tags must be a list of strings")
    return JudgeVerdict(
        verdict=verdict,
        weak_score=_optional_float(data.get("weak_score")),
        strong_score=_optional_float(data.get("strong_score")),
        gap=_optional_float(data.get("gap")),
        quality=_normalize_quality(data.get("quality")),
        reason=str(data["reason"]) if data.get("reason") else None,
        feedback=str(data["feedback"]) if data.get("feedback") else None,
        tags=[str(item) for item in tags if isinstance(item, str)],
        raw=dict(data),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    _validate_score(parsed, "score")
    return parsed


def _validate_score(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")


def _normalize_quality(value: Any) -> str | None:
    if value is None or value == "":
        return None
    quality = str(value).lower()
    _quality_rank(quality)
    return quality


def _quality_rank(value: str | None) -> int:
    quality = value or "medium"
    ranks = {"low": 0, "medium": 1, "high": 2}
    if quality not in ranks:
        raise ValueError("quality must be one of: low, medium, high")
    return ranks[quality]


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
