from __future__ import annotations

import json
import os
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
DestinationWriter = Callable[[Iterable[Mapping[str, Any]], Any], int]
PROVENANCE_KEYS = ("gap", "weak_score", "strong_score", "quality", "tags")
CHAT_TEMPLATES = ("chatml", "sharegpt")


@dataclass(frozen=True, slots=True)
class ExportOptions:
    conversational: bool = False
    chat_template: str = "sharegpt"


@dataclass(slots=True)
class ExportResult:
    records: int
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    notices: dict[str, int] = field(default_factory=dict)

    def summary(self, output: str | Path) -> str:
        message = f"exported {self.records} records to {output}"
        details: list[str] = []
        if self.skipped:
            reasons = ", ".join(
                f"{count} {reason}" for reason, count in sorted(self.skip_reasons.items())
            )
            details.append(f"skipped {self.skipped}: {reasons}")
        if self.notices:
            notices = ", ".join(
                f"{count} {reason}" for reason, count in sorted(self.notices.items())
            )
            details.append(f"notices: {notices}")
        if details:
            message += f" ({'; '.join(details)})"
        return message


@dataclass(frozen=True, slots=True)
class HFDestinationConfig:
    repo: str
    private: bool = False
    token: str | None = None
    card_metadata: Mapping[str, Any] = field(default_factory=dict)


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
    output: str | Path | None = None,
    conversational: bool = False,
    chat_template: str = "sharegpt",
    repo: str | None = None,
    private: bool = False,
    token: str | None = None,
    card_metadata: Mapping[str, Any] | None = None,
) -> ExportResult:
    chat_template = _validate_chat_template(chat_template)
    if conversational and format_name != "dpo":
        raise ValueError("--conversational is only supported with format dpo")
    if chat_template != "sharegpt" and format_name != "sft":
        raise ValueError("--chat-template is only supported with format sft")
    serializer = get_format(format_name)
    writer = get_destination(destination_name)
    stats = _ExportStats()
    options = ExportOptions(conversational=conversational, chat_template=chat_template)
    target = _destination_target(
        destination_name,
        output=output,
        repo=repo,
        private=private,
        token=token,
        card_metadata={"format": format_name, **dict(card_metadata or {})},
    )
    records = writer(_serialize_examples(examples, serializer, options, stats, format_name), target)
    return ExportResult(
        records=records,
        skipped=stats.skipped,
        skip_reasons=stats.skip_reasons,
        notices=stats.notices,
    )


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


def sft_serializer(options: ExportOptions, example: Example) -> Iterable[JSON]:
    assistant = _sft_assistant_output(example)
    system = _system_content(example)
    metadata = _sft_metadata(example)

    if options.chat_template == "chatml":
        messages = _sft_prompt_messages(example, system)
        messages.append({"role": "assistant", "content": assistant})
        yield {"messages": messages, "metadata": metadata}
        return

    conversations = _sharegpt_conversations(_sft_prompt_messages(example, system))
    conversations.append({"from": "gpt", "value": assistant})
    yield {"conversations": conversations, "metadata": metadata}


def write_local_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path) -> int:
    if isinstance(output, HFDestinationConfig):
        raise ValueError("local export requires --output")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_hf_dataset(records: Iterable[Mapping[str, Any]], output: Any) -> int:
    if not isinstance(output, HFDestinationConfig):
        raise ValueError("hf export requires --repo")
    token = _resolve_hf_token(output.token)
    hf_api = _load_hf_api()
    payload = list(records)
    jsonl = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in payload)
    card = _dataset_card(output, record_count=len(payload))
    api = hf_api(token=token)
    api.create_repo(
        repo_id=output.repo,
        repo_type="dataset",
        private=output.private,
        exist_ok=True,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=jsonl.encode("utf-8"),
        path_in_repo="data/train.jsonl",
        repo_id=output.repo,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=output.repo,
        repo_type="dataset",
        token=token,
    )
    return len(payload)


FORMAT_REGISTRY: dict[str, FormatSerializer] = {
    "dpo": dpo_serializer,
    "messages": messages_serializer,
    "prompt_completion": prompt_completion_serializer,
    "raw": raw_serializer,
    "sft": sft_serializer,
}
DESTINATION_REGISTRY: dict[str, DestinationWriter] = {
    "hf": write_hf_dataset,
    "local": write_local_jsonl,
}


def _serialize_examples(
    examples: Iterable[Example],
    serializer: FormatSerializer,
    options: ExportOptions,
    stats: "_ExportStats",
    format_name: str,
) -> Iterable[Mapping[str, Any]]:
    for example in examples:
        try:
            records = list(serializer(options, example))
        except SkipExample as exc:
            stats.skip(exc.reason)
            continue
        if format_name == "sft" and _uses_prompt_json_fallback(example):
            stats.notice("prompt JSON fallback")
        yield from records


def _destination_target(
    destination_name: str,
    *,
    output: str | Path | None,
    repo: str | None,
    private: bool,
    token: str | None,
    card_metadata: Mapping[str, Any],
) -> Any:
    if destination_name == "hf":
        if not repo:
            raise ValueError("--repo is required when using --to hf")
        return HFDestinationConfig(
            repo=repo,
            private=private,
            token=token,
            card_metadata=card_metadata,
        )
    if output is None:
        raise ValueError("--output is required when using --to local")
    return output


def _load_hf_api() -> type[Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError(
            "Hugging Face export requires optional dependencies. "
            "Install with: pip install datasmith[hf]"
        ) from exc
    return HfApi


def _load_hf_login_token() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError as exc:
        raise ValueError(
            "Hugging Face export requires optional dependencies. "
            "Install with: pip install datasmith[hf]"
        ) from exc
    return get_token()


def _resolve_hf_token(explicit_token: str | None) -> str:
    token = explicit_token or os.environ.get("HF_TOKEN") or _load_hf_login_token()
    if not token:
        raise ValueError("HF_TOKEN is required for --to hf. Set HF_TOKEN or run huggingface-cli login.")
    return token


def _dataset_card(config: HFDestinationConfig, *, record_count: int) -> str:
    metadata = dict(config.card_metadata)
    run_summary = metadata.get("run_summary")
    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "- text-generation",
        "---",
        "",
        "# DataSmith Dataset",
        "",
        "Generated by DataSmith.",
        "",
        "## Export",
        "",
        f"- format: `{metadata.get('format', 'unknown')}`",
        f"- records: {record_count}",
        f"- private: {str(config.private).lower()}",
    ]
    if metadata.get("source"):
        lines.append(f"- source: `{metadata['source']}`")
    if isinstance(run_summary, Mapping):
        lines.extend(
            [
                "",
                "## Run Provenance",
                "",
                f"- accepted: {run_summary.get('accepted', 'unknown')}",
                f"- rejected: {run_summary.get('rejected', 'unknown')}",
                f"- attempts: {run_summary.get('attempts', 'unknown')}",
                f"- average gap: {run_summary.get('avg_gap', 'unknown')}",
                f"- target count: {run_summary.get('target_count', 'unknown')}",
                f"- target met: {run_summary.get('target_met', 'unknown')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Acceptance Policy",
            "",
            "Acceptance policy thresholds are not recorded in current run summaries.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _sft_assistant_output(example: Example) -> str:
    if example.expected is not None:
        output = render_completion(example).strip()
        if output:
            return output
        raise SkipExample("no assistant output")
    try:
        return _first_attempt_output(example, "strong_attempts")
    except SkipExample as exc:
        raise SkipExample("no assistant output") from exc


def _dpo_metadata(example: Example) -> JSON:
    judge = example.metadata.get("judge")
    if not isinstance(judge, Mapping):
        return {}
    return {key: judge[key] for key in PROVENANCE_KEYS if key in judge}


def _sft_metadata(example: Example) -> JSON:
    metadata = _dpo_metadata(example)
    if "generator" in example.metadata:
        metadata["generator"] = example.metadata["generator"]
    return metadata


def _system_content(example: Example) -> str | None:
    value = example.metadata.get("system")
    if value in (None, ""):
        return None
    text = _stringify_prompt_value(value).strip()
    return text or None


def _sft_prompt_messages(example: Example, system: str | None) -> list[JSON]:
    messages = render_messages(example)
    if system is not None and not _has_system_message(messages):
        return [{"role": "system", "content": system}, *messages]
    return messages


def _has_system_message(messages: list[JSON]) -> bool:
    return bool(messages) and messages[0].get("role") == "system"


def _sharegpt_conversations(messages: list[JSON]) -> list[JSON]:
    role_map = {"assistant": "gpt", "system": "system", "user": "human"}
    return [
        {
            "from": role_map.get(str(message["role"]), str(message["role"])),
            "value": str(message["content"]),
        }
        for message in messages
    ]


def _uses_prompt_json_fallback(example: Example) -> bool:
    value = example.input
    if not isinstance(value, dict):
        return False
    return not any(key in value and value[key] not in (None, "") for key in DEFAULT_PROMPT_KEYS)


def _validate_chat_template(name: str) -> str:
    if name not in CHAT_TEMPLATES:
        raise ValueError(
            f"unknown chat template: {name}. Valid chat templates: {', '.join(CHAT_TEMPLATES)}"
        )
    return name


@dataclass(slots=True)
class _ExportStats:
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    notices: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def notice(self, reason: str) -> None:
        self.notices[reason] = self.notices.get(reason, 0) + 1


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
