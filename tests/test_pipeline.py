from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from asi import (
    AcceptancePolicy,
    AgenticSelfInstruct,
    DeterministicChallenger,
    DeterministicJudge,
    DeterministicSolver,
)
from asi.prompts import solver_prompt
from asi.types import Example


def test_direct_loop_accepts_when_gap_is_sufficient() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=DeterministicJudge(),
        weak_rollouts=1,
        strong_rollouts=1,
    )

    result = runner.run([Example(input={"seed": "trace"}, expected={"answer": "ok"})], target_count=2)

    assert len(result.accepted) == 2
    assert result.rejected == []
    assert result.accepted[0].metadata["judge"]["gap"] >= 0.2
    assert result.summary()["accepted"] == 2
    assert result.summary()["target_count"] == 2
    assert result.summary()["target_met"] is True


def test_rejections_include_reason_and_solver_outputs() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("strong"),
        strong_solver=DeterministicSolver("strong"),
        judge=DeterministicJudge(),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].reason_code == "judge_rejected"
    assert result.rejected[0].weak_attempts[0].output
    assert result.summary()["rejection_reasons"]["judge_rejected"] == 1


def test_policy_rejection_is_distinct_from_judge_rejection() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=_StaticJudge(
            {
                "verdict": "accept",
                "weak_score": 0.1,
                "strong_score": 0.95,
                "gap": 0.85,
                "quality": "medium",
                "tags": [],
            }
        ),
        policy=AcceptancePolicy(min_quality="high"),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "policy_rejected"


def test_summary_marks_partial_runs_when_target_is_not_met() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("strong"),
        strong_solver=DeterministicSolver("strong"),
        judge=DeterministicJudge(),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=2)

    assert len(result.accepted) == 0
    assert result.attempts == 2
    assert result.summary()["target_count"] == 2
    assert result.summary()["target_met"] is False


def test_solver_prompt_omits_expected_answer() -> None:
    prompt = solver_prompt(
        "weak",
        Example(input={"task": "solve me"}, expected={"answer": "secret reference"}),
    )

    assert "solve me" in prompt
    assert "secret reference" not in prompt
    assert "expected" not in prompt.lower()


def test_malformed_judge_output_is_rejected_instead_of_accepted() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=_StaticJudge(
            {
                "verdict": "accept",
                "weak_score": 0.1,
                "strong_score": 0.95,
                "gap": 0.85,
                "quality": "excellent",
                "tags": [],
            }
        ),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "judge_parse_error"


def test_non_finite_judge_scores_are_rejected() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=_StaticJudge(
            {
                "verdict": "accept",
                "weak_score": 0.1,
                "strong_score": 0.95,
                "gap": "nan",
                "quality": "high",
                "tags": [],
            }
        ),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "judge_parse_error"


def test_judge_tags_must_be_a_list() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=_StaticJudge(
            {
                "verdict": "accept",
                "weak_score": 0.1,
                "strong_score": 0.95,
                "gap": 0.85,
                "quality": "high",
                "tags": "reasoning",
            }
        ),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "judge_parse_error"


def test_challenger_provider_error_becomes_rejection() -> None:
    runner = AgenticSelfInstruct(
        challenger=_FailingModel("challenger unavailable"),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=DeterministicJudge(),
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "candidate_parse_error"
    assert "challenger unavailable" in result.rejected[0].reason


def test_judge_provider_error_becomes_rejection() -> None:
    runner = AgenticSelfInstruct(
        challenger=DeterministicChallenger(),
        weak_solver=DeterministicSolver("weak"),
        strong_solver=DeterministicSolver("strong"),
        judge=_FailingModel("judge unavailable"),
        weak_rollouts=1,
        strong_rollouts=1,
        max_attempts_per_example=1,
    )

    result = runner.run([Example(input={"seed": "trace"})], target_count=1)

    assert len(result.accepted) == 0
    assert result.rejected[0].reason_code == "judge_parse_error"
    assert "judge unavailable" in result.rejected[0].reason


def test_runner_rejects_invalid_rollout_configuration() -> None:
    with pytest.raises(ValueError, match="weak_rollouts"):
        AgenticSelfInstruct(
            challenger=DeterministicChallenger(),
            weak_solver=DeterministicSolver("weak"),
            strong_solver=DeterministicSolver("strong"),
            judge=DeterministicJudge(),
            weak_rollouts=0,
        )


class _StaticJudge:
    def __init__(self, verdict: Mapping[str, Any]) -> None:
        self.verdict = verdict

    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        return json.dumps(self.verdict)


class _FailingModel:
    def __init__(self, message: str) -> None:
        self.message = message

    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        raise RuntimeError(self.message)
