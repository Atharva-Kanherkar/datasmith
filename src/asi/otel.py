from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from asi.types import Example

PROMPT_KEYS = (
    "gen_ai.prompt",
    "gen_ai.input.messages",
    "llm.prompt",
    "openinference.input.value",
    "input.value",
    "input",
    "prompt",
)
COMPLETION_KEYS = (
    "gen_ai.completion",
    "gen_ai.output.messages",
    "llm.completion",
    "openinference.output.value",
    "output.value",
    "output",
    "completion",
    "response",
)


def examples_from_otlp(path_or_data: str | Path | Mapping[str, Any]) -> list[Example]:
    data = _load_json(path_or_data)
    spans = list(_iter_otlp_spans(data))
    return [_span_to_example(span) for span in spans]


def examples_from_span_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(span, dict):
                raise ValueError(f"{path}:{line_number}: span record must be an object")
            examples.append(_span_to_example(span))
    return examples


def _load_json(path_or_data: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(path_or_data, Mapping):
        return path_or_data
    with Path(path_or_data).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("OTLP JSON root must be an object")
    return data


def _iter_otlp_spans(data: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for resource_span in data.get("resourceSpans", []):
        resource_attrs = _attrs_to_dict(resource_span.get("resource", {}).get("attributes", []))
        scope_spans = resource_span.get("scopeSpans") or resource_span.get("instrumentationLibrarySpans") or []
        for scope_span in scope_spans:
            scope_attrs = _attrs_to_dict(scope_span.get("scope", {}).get("attributes", []))
            for span in scope_span.get("spans", []):
                merged = dict(span)
                metadata = {
                    "resource": resource_attrs,
                    "scope": scope_attrs,
                    "trace_id": span.get("traceId") or span.get("trace_id"),
                    "span_id": span.get("spanId") or span.get("span_id"),
                    "name": span.get("name"),
                }
                attrs = _attrs_to_dict(span.get("attributes", []))
                metadata["attributes"] = attrs
                merged["metadata"] = metadata
                merged["attributes"] = attrs
                yield merged


def _span_to_example(span: Mapping[str, Any]) -> Example:
    attributes = _extract_attrs(span)
    prompt = _first_present(attributes, PROMPT_KEYS)
    completion = _first_present(attributes, COMPLETION_KEYS)
    trace_id = span.get("traceId") or span.get("trace_id") or span.get("metadata", {}).get("trace_id")
    span_id = span.get("spanId") or span.get("span_id") or span.get("metadata", {}).get("span_id")
    name = span.get("name") or span.get("metadata", {}).get("name")
    if prompt is None:
        prompt = {
            "operation": name or attributes.get("gen_ai.operation.name") or "otel_span",
            "attributes": attributes,
        }
    metadata = {
        "source": "otel",
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "attributes": attributes,
    }
    if isinstance(span.get("metadata"), dict):
        metadata.update({k: v for k, v in span["metadata"].items() if k not in metadata})
    return Example(input=prompt, expected=completion, metadata=metadata)


def _extract_attrs(span: Mapping[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes", {})
    if isinstance(raw, list):
        return _attrs_to_dict(raw)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    if not isinstance(attrs, list):
        return {}
    output: dict[str, Any] = {}
    for attr in attrs:
        if not isinstance(attr, dict) or "key" not in attr:
            continue
        output[str(attr["key"])] = _otel_value(attr.get("value"))
    return output


def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        return [_otel_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attrs_to_dict(value["kvlistValue"].get("values", []))
    return value


def _first_present(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return value
    return None
