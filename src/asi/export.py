from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asi.types import Example, JSON


DEFAULT_PROMPT_KEYS = (
    "gen_ai.prompt",
    "gen_ai.input.messages",
    "llm.prompt",
    "openinference.input.value",
    "input.value",
    "input",
    "prompt",
)
FormatSerializer = Callable[["ExportOptions", Example], Iterable[Mapping[str, Any]]]
DestinationWriter = Callable[[Iterable[Mapping[str, Any]], str | Path], int]
PROVENANCE_KEYS = ("gap", "weak_score", "strong_score", "quality", "tags")


@dataclass(frozen=True, slots=True)
class ExportOptions:
    conversational: bool = False


@dataclass(slots=True)
class ExportResult:
    records: int
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def summary(self, output: str | Path) -> str:
        message = f"exported {self.records} records to {output}"
        if self.skipped:
            reasons = ", ".join(
                f"{count} {reason}" for reason, count in sorted(self.skip_reasons.items())
            )
            message += f" (skipped {self.skipped}: {reasons})"
        return message


class SkipExample(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def render_prompt(
    example: Example,
    *,
    prompt_keys: Sequence[str] = DEFAULT_PROMPT_KEYS,
) -> str:
    """Return a trainer prompt string from an example input."""
    value = example.input
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in prompt_keys:
            if key in value and value[key] not in (None, ""):
                return _stringify_prompt_value(value[key])
        return _compact_json(value)
    return _stringify_prompt_value(value)


def render_messages(example: Example) -> list[JSON]:
    """Return trainer messages, preserving existing role/content arrays when present."""
    value = example.input
    if isinstance(value, list) and _is_message_list(value):
        return [dict(message) for message in value]
    if isinstance(value, dict):
        for key in DEFAULT_PROMPT_KEYS:
            nested = value.get(key)
            if isinstance(nested, list) and _is_message_list(nested):
                return [dict(message) for message in nested]
    return [{"role": "user", "content": render_prompt(example)}]


def export_examples(
    examples: Iterable[Example],
    *,
    format_name: str,
    destination_name: str,
    output: str | Path,
    conversational: bool = False,
) -> ExportResult:
    if conversational and format_name != "dpo":
        raise ValueError("--conversational is only supported with format dpo")
    serializer = get_format(format_name)
    writer = get_destination(destination_name)
    stats = _ExportStats()
    options = ExportOptions(conversational=conversational)
    records = writer(_serialize_examples(examples, serializer, options, stats), output)
    return ExportResult(records=records, skipped=stats.skipped, skip_reasons=stats.skip_reasons)


def export_formats() -> list[str]:
    return sorted(FORMAT_REGISTRY)


def export_destinations() -> list[str]:
    return sorted(DESTINATION_REGISTRY)


def get_format(name: str) -> FormatSerializer:
    try:
        return FORMAT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown export format: {name}. Valid formats: {', '.join(export_formats())}"
        ) from exc


def get_destination(name: str) -> DestinationWriter:
    try:
        return DESTINATION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown export destination: {name}. Valid destinations: "
            f"{', '.join(export_destinations())}"
        ) from exc


def raw_serializer(options: ExportOptions, example: Example) -> Iterable[JSON]:
    yield example.to_dict()


def prompt_completion_serializer(options: ExportOptions, example: Example) -> Iterable[JSON]:
    if example.expected is None:
        raise ValueError("prompt_completion export requires examples with expected outputs")
    yield {
        "prompt": render_prompt(example),
        "completion": render_completion(example),
    }


def messages_serializer(options: ExportOptions, example: Example) -> Iterable[JSON]:
    if example.expected is None:
        raise ValueError("messages export requires examples with expected outputs")
    yield {
        "messages": [
            {"role": "user", "content": render_prompt(example)},
            {"role": "assistant", "content": render_completion(example)},
        ]
    }


def dpo_serializer(options: ExportOptions, example: Example) -> Iterable[JSON]:
    prompt = render_prompt(example)
    chosen = _first_attempt_output(example, "strong_attempts")
    rejected = _first_attempt_output(example, "weak_attempts")
    if chosen == rejected:
        raise SkipExample("identical chosen/rejected")

    if options.conversational:
        yield {
            "prompt": render_messages(example),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "metadata": _dpo_metadata(example),
        }
        return

    yield {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": _dpo_metadata(example),
    }


def write_local_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path) -> int:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


FORMAT_REGISTRY: dict[str, FormatSerializer] = {
    "dpo": dpo_serializer,
    "messages": messages_serializer,
    "prompt_completion": prompt_completion_serializer,
    "raw": raw_serializer,
}
DESTINATION_REGISTRY: dict[str, DestinationWriter] = {
    "local": write_local_jsonl,
}


def _serialize_examples(
    examples: Iterable[Example],
    serializer: FormatSerializer,
    options: ExportOptions,
    stats: "_ExportStats",
) -> Iterable[Mapping[str, Any]]:
    for example in examples:
        try:
            records = list(serializer(options, example))
        except SkipExample as exc:
            stats.skip(exc.reason)
            continue
        yield from records


def _stringify_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return _compact_json(value)
    return str(value)


def _is_message_list(value: list[Any]) -> bool:
    return all(
        isinstance(item, Mapping)
        and isinstance(item.get("role"), str)
        and isinstance(item.get("content"), str)
        for item in value
    )


def render_completion(example: Example) -> str:
    """Return a trainer completion string from an example expected output."""
    return _stringify_prompt_value(example.expected)


def _first_attempt_output(example: Example, key: str) -> str:
    attempts = example.metadata.get(key)
    if not isinstance(attempts, list) or not attempts:
        raise SkipExample("no solver attempts")
    first = attempts[0]
    if not isinstance(first, Mapping) or first.get("output") in (None, ""):
        raise SkipExample("no solver attempts")
    output = str(first["output"]).strip()
    if not output:
        raise SkipExample("no solver attempts")
    return output


def _dpo_metadata(example: Example) -> JSON:
    judge = example.metadata.get("judge")
    if not isinstance(judge, Mapping):
        return {}
    return {key: judge[key] for key in PROVENANCE_KEYS if key in judge}


@dataclass(slots=True)
class _ExportStats:
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
