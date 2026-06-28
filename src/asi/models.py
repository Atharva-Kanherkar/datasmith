from __future__ import annotations

import json
from typing import Any, Mapping


class DeterministicChallenger:
    def __init__(self) -> None:
        self.count = 0

    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        self.count += 1
        return json.dumps(
            {
                "input": {
                    "task": f"Explain the failure mode in trace cluster {self.count}",
                    "context": "A weak model tends to stop at the first plausible answer.",
                },
                "expected": {
                    "answer": "Identify the hidden constraint, compare alternatives, and justify the final answer."
                },
                "metadata": {"source": "deterministic-demo", "role": role},
            },
            sort_keys=True,
        )


class DeterministicSolver:
    def __init__(self, strength: str) -> None:
        if strength not in {"weak", "strong"}:
            raise ValueError("strength must be weak or strong")
        self.strength = strength

    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        if "\"expected\"" in prompt or "Reference answer" in prompt:
            raise AssertionError("solver prompt leaked expected answer")
        if self.strength == "weak":
            return "I would answer with the most obvious explanation, but I may miss hidden constraints."
        return "The key is to inspect the hidden constraint, compare alternatives, and justify the final answer."


class DeterministicJudge:
    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        weak_score = 0.35 if "miss hidden constraints" in prompt else 0.65
        strong_score = 0.85 if "hidden constraint" in prompt else 0.55
        gap = max(0.0, strong_score - weak_score)
        accepted = gap >= 0.2 and weak_score <= 0.5 and strong_score >= 0.65
        return json.dumps(
            {
                "verdict": "accept" if accepted else "reject",
                "weak_score": weak_score,
                "strong_score": strong_score,
                "gap": gap,
                "quality": "high" if accepted else "medium",
                "reason": "clear weak/strong separation" if accepted else "insufficient gap",
                "feedback": "Make weak failure modes depend on hidden constraints.",
                "tags": ["reasoning", "trace-analysis"],
            },
            sort_keys=True,
        )
