from __future__ import annotations

from asi import AgenticSelfInstruct, DeterministicChallenger, DeterministicJudge, DeterministicSolver
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
    assert result.rejected[0].reason_code == "quality_rejected"
    assert result.rejected[0].weak_attempts[0].output
    assert result.summary()["rejection_reasons"]["quality_rejected"] == 1


def test_solver_prompt_omits_expected_answer() -> None:
    prompt = solver_prompt(
        "weak",
        Example(input={"task": "solve me"}, expected={"answer": "secret reference"}),
    )

    assert "solve me" in prompt
    assert "secret reference" not in prompt
    assert "expected" not in prompt.lower()
