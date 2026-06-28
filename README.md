# Agentic Self-Instruct

Standalone, provider-agnostic SDK and CLI for building synthetic training/eval data with the
Agentic Self-Instruct loop described in Meta FAIR's Autodata paper.

This project is not tied to AgentClash. It is a small OSS implementation that lets developers bring
their own models, traces, seeds, and evaluation workflow.

## Why this exists

Classic Self-Instruct asks a model to create data. Agentic Self-Instruct adds a useful pressure test:

1. A challenger creates a candidate example.
2. A weak solver attempts it.
3. A strong solver attempts it.
4. A judge accepts examples where the strong solver succeeds and the weak solver struggles.
5. Judge feedback is fed into later challenger attempts.

The result is a dataset aimed at teaching or evaluating the weak model on examples that are neither
trivial nor impossible.

## Install

```bash
pip install agentic-self-instruct
```

For local development:

```bash
git clone https://github.com/Atharva-Kanherkar/agentic-self-instruct
cd agentic-self-instruct
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Quickstart

Run the deterministic demo with no API keys:

```bash
asi run --seeds examples/seeds.jsonl --output-dir runs/demo --target-count 2 --local-demo
```

Convert OTLP JSON traces to seed examples:

```bash
asi ingest-otel examples/otel-traces.json --output runs/otel-seeds.jsonl
```

Use the SDK directly:

```python
from asi import AgenticSelfInstruct, DeterministicChallenger, DeterministicJudge, DeterministicSolver
from asi.io import read_jsonl

runner = AgenticSelfInstruct(
    challenger=DeterministicChallenger(),
    weak_solver=DeterministicSolver("weak"),
    strong_solver=DeterministicSolver("strong"),
    judge=DeterministicJudge(),
)

result = runner.run(read_jsonl("examples/seeds.jsonl"), target_count=2)
print(result.summary())
```

## Bring Your Own Models

Any object with this method can be used:

```python
class MyModel:
    def complete(self, prompt: str, *, role: str, metadata: dict) -> str:
        ...
```

Use different implementations for challenger, weak solver, strong solver, and judge. The included
`OpenAICompatibleModel` is optional and dependency-free:

```python
from asi.providers import OpenAICompatibleModel

model = OpenAICompatibleModel(model="gpt-4.1-mini")
```

## OpenTelemetry Data

The package supports OTLP JSON exports and flattened JSONL span records. It preserves:

- `trace_id`, `span_id`, span name, resource attributes, and scope attributes
- GenAI/OpenInference-style prompt and completion attributes
- All original span attributes in example metadata

Preferred prompt attributes include `gen_ai.prompt`, `gen_ai.input.messages`, `llm.prompt`,
`openinference.input.value`, `input.value`, and `prompt`. Completion attributes follow the same
pattern with output/completion names.

See [docs/otel.md](docs/otel.md).

## Research

This implementation follows the practical Agentic Self-Instruct loop from Meta FAIR's Autodata
paper, arXiv:2606.25996v2. It intentionally does not reproduce Meta's full training or
meta-optimization stack. Instead it provides the reusable developer substrate: ingestion, model
interfaces, loop orchestration, acceptance policies, artifacts, CLI, and tests.

See [docs/research.md](docs/research.md).

## Status

Alpha. The public API is small, typed, and tested, but expect iteration as the paper ecosystem
matures.

## License

MIT.
