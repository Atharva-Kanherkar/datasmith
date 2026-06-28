from __future__ import annotations

import json
from typing import Any, Mapping

from asi.seed_constructor import WebSearchSignal


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


class DeterministicSearchClient:
    def search(self, query: str, *, limit: int) -> list[WebSearchSignal]:
        signals = [
            WebSearchSignal(
                title=f"{query} operational signal",
                url="https://example.com/datasmith/operational-signal",
                snippet=(
                    "Real systems often fail when an agent follows the first plausible answer "
                    "without checking the hidden constraint or latest state."
                ),
            ),
            WebSearchSignal(
                title=f"{query} policy exception signal",
                url="https://example.com/datasmith/policy-exception",
                snippet=(
                    "Useful evaluation examples include an exception, a fact pattern, and a "
                    "clear reason why the exception controls the outcome."
                ),
            ),
        ]
        return signals[:limit]


class DeterministicSeedConstructor:
    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        domain = str(metadata.get("domain") or "general reasoning")
        return json.dumps(
            {
                "input": {
                    "task": f"Apply a grounded rule in {domain}.",
                    "context": (
                        "A weak model may choose the obvious answer before checking the "
                        "exception described in the source signal."
                    ),
                    "question": "Which hidden condition changes the final answer?",
                },
                "expected": {
                    "answer": (
                        "Identify the exception or freshest state first, then apply it before "
                        "making the final decision."
                    )
                },
                "metadata": {
                    "domain": domain,
                    "source": "deterministic-seed-constructor",
                    "signals": ["operational-signal", "policy-exception"],
                },
            },
            sort_keys=True,
        )


class DeterministicSeedJudge:
    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        return json.dumps(
            {
                "verdict": "accept",
                "score": 0.86,
                "reason": "seed is grounded, specific, and contains an exception pattern",
                "feedback": "Keep examples tied to concrete hidden constraints.",
                "tags": ["grounded", "exception", "seed"],
            },
            sort_keys=True,
        )
