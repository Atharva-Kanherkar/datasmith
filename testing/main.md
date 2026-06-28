# DataSmith OSS SDK — Test Contract

## Functional Behavior
- The repository is a standalone, OSS Python package and CLI, independent of AgentClash.
- The SDK implements a DataSmith seed-construction stage:
  - A seed-constructor model creates initial seeds from a domain brief and web-search signals.
  - A seed judge scores seeds for grounding, specificity, and usefulness.
  - Web-search access is isolated to seed construction; downstream models consume artifacts.
- The SDK implements the paper-inspired weak-vs-strong Agentic Self-Instruct loop:
  - A challenger model proposes examples from seeds/context.
  - Weak and strong solver models attempt each candidate without seeing the expected answer.
  - A judge scores weak/strong outputs, quality, gap, and acceptance.
  - Accepted examples, rejected attempts, metrics, and recipe feedback are returned as structured artifacts.
- The SDK is provider-agnostic:
  - Developers can pass any model callable implementing the SDK protocol.
  - A deterministic local model is available for tests and examples.
  - Optional OpenAI-compatible HTTP usage is documented but not required for tests.
- OTEL-shaped data ingestion is first-class:
  - OTLP JSON trace exports can be loaded as seed examples.
  - JSONL span/event files can be loaded as seed examples.
  - Semantic attributes such as prompts, completions, tool names, status, trace IDs, and span IDs are preserved in metadata.
- The CLI supports:
  - `datasmith construct-seeds` for constructing initial seed JSONL from a domain brief.
  - `datasmith run` for generation from normalized JSONL seed files.
  - `datasmith ingest-otel` for converting OTEL-shaped traces into normalized seed JSONL.
  - `datasmith init` for creating a starter config scaffold. The current CLI run path is demo-only until
    provider/config loading is added.
- The repo includes complete OSS basics: README, license, contributing guide, code of conduct, security policy, examples, tests, pyproject, and package exports.

## Unit Tests
- `test_otel_ingestion.py`:
  - OTLP resource spans are converted into seeds with trace/span metadata.
  - JSONL span records are converted into seeds.
  - Prompt/completion attributes are preferred when present.
- `test_pipeline.py`:
  - The direct weak/strong loop accepts candidates when the judge sees sufficient gap.
  - Rejections include reason codes and solver outputs.
  - Solver prompts do not include expected answers.
- `test_io.py`:
  - JSONL read/write round-trips examples.
  - Run result exports include accepted and rejected records.
- `test_seed_constructor.py`:
  - Seed construction uses search signals and accepts grounded seeds.
  - Invalid seed-construction target counts are rejected.
- `test_cli.py`:
  - `datasmith ingest-otel` writes normalized JSONL.
  - `datasmith construct-seeds` writes seed-construction artifacts.
  - `datasmith run` works with deterministic local models and writes output files.

## Integration / Functional Tests
- `python -m pytest`
- `python -m asi.cli --help`
- `python -m asi.cli init --output /tmp/datasmith-config.yaml`
- `python -m asi.cli construct-seeds --domain "legal refunds" --output-dir /tmp/datasmith-seeds --target-count 2 --local-demo`
- `python -m asi.cli ingest-otel examples/otel-traces.json --output /tmp/datasmith-otel-seeds.jsonl`
- `python -m asi.cli run --seeds examples/seeds.jsonl --output-dir /tmp/datasmith-run --target-count 2 --local-demo`

## Smoke Tests
- Install editable package with dev extras.
- Run the CLI on included examples without API keys.
- Verify generated artifacts contain `accepted.jsonl`, `rejected.jsonl`, and `summary.json`.

## E2E Tests
- N/A — this is an SDK/CLI package. End-to-end behavior is covered by CLI functional smoke tests.

## Manual / cURL Tests
- N/A — this package has no server component.

## Research Notes
- Meta Autodata / Agentic Self-Instruct paper: `2606.25996v2.pdf`.
- Core paper behavior implemented here: the inner Agentic Self-Instruct loop with challenger, weak
  solver, strong solver, judge, acceptance by weak/strong score gap, and feedback for later rounds.
  The full data-scientist outer loop and meta-optimization stack are out of scope for this package.
- The OSS package should avoid AgentClash-specific APIs and keep AgentClash as only an optional integration target in future work.
