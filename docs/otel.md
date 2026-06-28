# OpenTelemetry Ingestion

DataSmith supports two trace-like input formats.

## OTLP JSON

Use an OTLP JSON trace export with `resourceSpans`, `scopeSpans`, and `spans`:

```bash
datasmith ingest-otel examples/otel-traces.json --output runs/seeds.jsonl
```

The loader preserves resource attributes, scope attributes, span attributes, `traceId`, `spanId`,
and span name in each seed's metadata.

## Flattened Span JSONL

You can also provide one span object per line:

```json
{"trace_id":"t1","span_id":"s1","name":"chat","attributes":{"gen_ai.prompt":"...","gen_ai.completion":"..."}}
```

Convert with:

```bash
datasmith ingest-otel spans.jsonl --format jsonl --output runs/seeds.jsonl
```

## Attribute Extraction

Prompt-like attributes are preferred in this order:

- `gen_ai.prompt`
- `gen_ai.input.messages`
- `llm.prompt`
- `openinference.input.value`
- `input.value`
- `input`
- `prompt`

Completion-like attributes are preferred in this order:

- `gen_ai.completion`
- `gen_ai.output.messages`
- `llm.completion`
- `openinference.output.value`
- `output.value`
- `output`
- `completion`
- `response`

If no prompt-like attribute exists, the loader builds an input object from the span operation and
attributes.
