from __future__ import annotations

import json

from asi.otel import examples_from_otlp, examples_from_span_jsonl


def test_otlp_resource_spans_are_converted_to_examples() -> None:
    data = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "agent-api"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"attributes": []},
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "s1",
                                "name": "gen_ai.chat",
                                "attributes": [
                                    {
                                        "key": "gen_ai.prompt",
                                        "value": {"stringValue": "What failed?"},
                                    },
                                    {
                                        "key": "gen_ai.completion",
                                        "value": {"stringValue": "Tool timeout."},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    examples = examples_from_otlp(data)

    assert len(examples) == 1
    assert examples[0].input == "What failed?"
    assert examples[0].expected == "Tool timeout."
    assert examples[0].metadata["trace_id"] == "t1"
    assert examples[0].metadata["resource"]["service.name"] == "agent-api"


def test_span_jsonl_is_converted_to_examples(tmp_path) -> None:
    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps(
            {
                "trace_id": "trace-jsonl",
                "span_id": "span-jsonl",
                "name": "chat",
                "attributes": {
                    "prompt": "Find the hidden constraint",
                    "response": "Refresh state first",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = examples_from_span_jsonl(path)

    assert examples[0].input == "Find the hidden constraint"
    assert examples[0].expected == "Refresh state first"
    assert examples[0].metadata["span_id"] == "span-jsonl"
