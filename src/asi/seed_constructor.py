from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from asi.types import Example, Model


@dataclass(slots=True)
class WebSearchSignal:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SearchClient(Protocol):
    """Search capability reserved for seed construction."""

    def search(self, query: str, *, limit: int) -> list[WebSearchSignal]:
        """Return web search signals for a seed-construction query."""


@dataclass(slots=True)
class SeedJudgeVerdict:
    verdict: str
    score: float
    reason: str
    feedback: str | None = None
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def accepted(self) -> bool:
        return self.verdict == "accept"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RejectedSeed:
    example: Example
    reason_code: str
    reason: str
    judge: SeedJudgeVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "judge": self.judge.to_dict() if self.judge else None,
        }


@dataclass(slots=True)
class SeedConstructionResult:
    accepted: list[Example] = field(default_factory=list)
    rejected: list[RejectedSeed] = field(default_factory=list)
    signals: list[WebSearchSignal] = field(default_factory=list)
    attempts: int = 0
    target_count: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "attempts": self.attempts,
            "target_count": self.target_count,
            "target_met": len(self.accepted) >= self.target_count,
            "signals": len(self.signals),
            "rejection_reasons": _reason_counts(self.rejected),
        }


@dataclass(slots=True)
class SeedConstructor:
    """Constructs initial seed examples from a domain brief and search signals."""

    search_client: SearchClient
    constructor: Model
    judge: Model
    min_score: float = 0.7
    search_limit: int = 5
    max_attempts_per_seed: int = 4

    def __post_init__(self) -> None:
        if self.min_score < 0 or self.min_score > 1:
            raise ValueError("min_score must be between 0 and 1")
        if self.search_limit < 1:
            raise ValueError("search_limit must be at least 1")
        if self.max_attempts_per_seed < 1:
            raise ValueError("max_attempts_per_seed must be at least 1")

    def run(
        self,
        *,
        domain: str,
        target_count: int,
        queries: list[str] | None = None,
    ) -> SeedConstructionResult:
        if not domain.strip():
            raise ValueError("domain is required")
        if target_count < 1:
            raise ValueError("target_count must be at least 1")

        search_queries = queries or [domain]
        signals = self._collect_signals(search_queries)
        result = SeedConstructionResult(signals=signals, target_count=target_count)
        feedback: list[str] = []
        max_attempts = target_count * self.max_attempts_per_seed

        for attempt in range(1, max_attempts + 1):
            if len(result.accepted) >= target_count:
                break
            result.attempts = attempt
            example, error = self._construct_seed(domain, signals, feedback)
            if error:
                result.rejected.append(
                    RejectedSeed(
                        example=Example(input={"seed_error": True}),
                        reason_code="seed_parse_error",
                        reason=error,
                    )
                )
                continue

            verdict, judge_error = self._judge_seed(domain, example, signals)
            if judge_error:
                result.rejected.append(
                    RejectedSeed(
                        example=example,
                        reason_code="seed_judge_parse_error",
                        reason=judge_error,
                    )
                )
                continue

            if verdict.accepted() and verdict.score >= self.min_score:
                example.metadata = {
                    **example.metadata,
                    "generator": "datasmith-seed-constructor",
                    "seed_judge": verdict.to_dict(),
                    "web_signals": [signal.to_dict() for signal in signals],
                }
                result.accepted.append(example)
            else:
                reason = verdict.reason or "seed judge rejected example"
                feedback.append(verdict.feedback or reason)
                result.rejected.append(
                    RejectedSeed(
                        example=example,
                        reason_code="seed_judge_rejected",
                        reason=reason,
                        judge=verdict,
                    )
                )

        return result

    def _collect_signals(self, queries: list[str]) -> list[WebSearchSignal]:
        seen: set[str] = set()
        signals: list[WebSearchSignal] = []
        for query in queries:
            for signal in self.search_client.search(query, limit=self.search_limit):
                key = signal.url or f"{signal.title}:{signal.snippet}"
                if key in seen:
                    continue
                seen.add(key)
                signals.append(signal)
        return signals

    def _construct_seed(
        self, domain: str, signals: list[WebSearchSignal], feedback: list[str]
    ) -> tuple[Example, str | None]:
        try:
            raw = self.constructor.complete(
                _constructor_prompt(domain, signals, feedback),
                role="seed_constructor",
                metadata={"domain": domain, "signals": len(signals)},
            )
            parsed = json.loads(_strip_fences(raw))
            if not isinstance(parsed, dict):
                return Example(input={}), "seed constructor returned non-object JSON"
            return Example.from_dict(parsed), None
        except Exception as exc:  # noqa: BLE001
            return Example(input={}), f"parse seed constructor response: {exc}"

    def _judge_seed(
        self, domain: str, example: Example, signals: list[WebSearchSignal]
    ) -> tuple[SeedJudgeVerdict, str | None]:
        try:
            raw = self.judge.complete(
                _seed_judge_prompt(domain, example, signals),
                role="seed_judge",
                metadata={"domain": domain, "signals": len(signals)},
            )
            parsed = json.loads(_strip_fences(raw))
            if not isinstance(parsed, dict):
                return SeedJudgeVerdict(verdict="reject", score=0, reason=""), (
                    "seed judge returned non-object JSON"
                )
            return _seed_verdict_from_dict(parsed), None
        except Exception as exc:  # noqa: BLE001
            return SeedJudgeVerdict(verdict="reject", score=0, reason=""), (
                f"parse seed judge response: {exc}"
            )


def _constructor_prompt(
    domain: str, signals: list[WebSearchSignal], feedback: list[str]
) -> str:
    lines = [
        "You are the seed-constructor agent for DataSmith.",
        "Use the web-search signals to create one realistic seed example as strict JSON:",
        '{"input": {...}, "expected": {...}, "metadata": {"domain": "", "signals": []}}',
        "The seed should be grounded, specific, and useful for later synthetic data generation.",
        "",
        f"Domain brief: {domain}",
        "",
        "Web-search signals:",
    ]
    for index, signal in enumerate(signals, start=1):
        lines.append(f"{index}. {signal.title} ({signal.url})")
        lines.append(f"   {signal.snippet}")
    if feedback:
        lines.append("")
        lines.append("Previous seed judge feedback:")
        lines.extend(f"- {item}" for item in feedback[-6:])
    return "\n".join(lines)


def _seed_judge_prompt(domain: str, example: Example, signals: list[WebSearchSignal]) -> str:
    return "\n".join(
        [
            "You are the seed judge for DataSmith.",
            "Score whether this seed is grounded, realistic, specific, and useful.",
            "Return only JSON with this shape:",
            '{"verdict":"accept|reject","score":0.0,"reason":"","feedback":"","tags":[]}',
            "",
            f"Domain brief: {domain}",
            "",
            "Candidate seed:",
            json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True),
            "",
            "Available web-search signals:",
            json.dumps(
                [signal.to_dict() for signal in signals],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ]
    )


def _seed_verdict_from_dict(data: Mapping[str, Any]) -> SeedJudgeVerdict:
    verdict = str(data.get("verdict") or "reject")
    if verdict not in {"accept", "reject"}:
        raise ValueError("seed judge verdict must be accept or reject")
    score = float(data.get("score", 0))
    if score < 0 or score > 1:
        raise ValueError("seed judge score must be between 0 and 1")
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("seed judge tags must be a list")
    return SeedJudgeVerdict(
        verdict=verdict,
        score=score,
        reason=str(data["reason"]) if data.get("reason") else "",
        feedback=str(data["feedback"]) if data.get("feedback") else None,
        tags=[str(tag) for tag in tags if isinstance(tag, str)],
        raw=dict(data),
    )


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json")
    elif text.startswith("```"):
        text = text.removeprefix("```")
    if text.endswith("```"):
        text = text.removesuffix("```")
    return text.strip()


def _reason_counts(rejected: list[RejectedSeed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        counts[item.reason_code] = counts.get(item.reason_code, 0) + 1
    return counts
