from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
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
FormatSerializer = Callable[[Example], Iterable[Mapping[str, Any]]]
DestinationWriter = Callable[[Iterable[Mapping[str, Any]], str | Path], int]


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


def export_examples(
    examples: Iterable[Example],
    *,
    format_name: str,
    destination_name: str,
    output: str | Path,
) -> int:
    serializer = get_format(format_name)
    writer = get_destination(destination_name)
    return writer(_serialize_examples(examples, serializer), output)


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


def raw_serializer(example: Example) -> Iterable[JSON]:
    yield example.to_dict()


def prompt_completion_serializer(example: Example) -> Iterable[JSON]:
    if example.expected is None:
        raise ValueError("prompt_completion export requires examples with expected outputs")
    yield {
        "prompt": render_prompt(example),
        "completion": render_completion(example),
    }


def messages_serializer(example: Example) -> Iterable[JSON]:
    if example.expected is None:
        raise ValueError("messages export requires examples with expected outputs")
    yield {
        "messages": [
            {"role": "user", "content": render_prompt(example)},
            {"role": "assistant", "content": render_completion(example)},
        ]
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
) -> Iterable[Mapping[str, Any]]:
    for example in examples:
        yield from serializer(example)


def _stringify_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return _compact_json(value)
    return str(value)


def render_completion(example: Example) -> str:
    """Return a trainer completion string from an example expected output."""
    return _stringify_prompt_value(example.expected)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
