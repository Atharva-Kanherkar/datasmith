from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol


JSON = dict[str, Any]


class Model(Protocol):
    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        """Return a text completion for a prompt."""


@dataclass(slots=True)
class Example:
    input: Any
    expected: Any | None = None
    metadata: JSON = field(default_factory=dict)

    def to_dict(self) -> JSON:
        data: JSON = {"input": self.input, "metadata": self.metadata}
        if self.expected is not None:
            data["expected"] = self.expected
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Example":
        if "input" not in data:
            raise ValueError("example is missing required field: input")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("example metadata must be an object")
        return cls(input=data["input"], expected=data.get("expected"), metadata=dict(metadata))


@dataclass(slots=True)
class SolverAttempt:
    role: str
    output: str
    attempt: int
    metadata: JSON = field(default_factory=dict)

    def to_dict(self) -> JSON:
        return asdict(self)


@dataclass(slots=True)
class JudgeVerdict:
    verdict: str
    weak_score: float | None = None
    strong_score: float | None = None
    gap: float | None = None
    quality: str | None = None
    reason: str | None = None
    feedback: str | None = None
    tags: list[str] = field(default_factory=list)
    raw: JSON = field(default_factory=dict)

    def accepted(self) -> bool:
        return self.verdict == "accept"

    def to_dict(self) -> JSON:
        return asdict(self)


@dataclass(slots=True)
class RejectedExample:
    candidate: Example
    reason_code: str
    reason: str
    weak_attempts: list[SolverAttempt] = field(default_factory=list)
    strong_attempts: list[SolverAttempt] = field(default_factory=list)
    judge: JudgeVerdict | None = None

    def to_dict(self) -> JSON:
        return {
            "candidate": self.candidate.to_dict(),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "weak_attempts": [attempt.to_dict() for attempt in self.weak_attempts],
            "strong_attempts": [attempt.to_dict() for attempt in self.strong_attempts],
            "judge": self.judge.to_dict() if self.judge else None,
        }


@dataclass(slots=True)
class RunResult:
    accepted: list[Example] = field(default_factory=list)
    rejected: list[RejectedExample] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    attempts: int = 0
    target_count: int | None = None

    def summary(self) -> JSON:
        accepted_scores = [
            item.metadata.get("judge", {}) for item in self.accepted if isinstance(item.metadata, dict)
        ]
        gaps = [score.get("gap") for score in accepted_scores if isinstance(score.get("gap"), int | float)]
        target_met = len(self.accepted) >= self.target_count if self.target_count is not None else None
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "attempts": self.attempts,
            "target_count": self.target_count,
            "target_met": target_met,
            "avg_gap": sum(gaps) / len(gaps) if gaps else None,
            "rejection_reasons": _reason_counts(self.rejected),
            "feedback": self.feedback,
        }


def _reason_counts(rejected: list[RejectedExample]) -> JSON:
    counts: JSON = {}
    for item in rejected:
        counts[item.reason_code] = int(counts.get(item.reason_code, 0)) + 1
    return counts
