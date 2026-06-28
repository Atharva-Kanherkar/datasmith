# codex/dpo-export-preset — Test Contract

## Functional Behavior

- `datasmith export --from <dir> --format dpo --to local --output dpo.jsonl` reads `<dir>/accepted.jsonl` and writes one JSONL record per valid accepted example.
- `datasmith export --from <accepted.jsonl> --format dpo --to local --output dpo.jsonl` accepts a direct JSONL path.
- Each standard DPO record contains `prompt`, `chosen`, `rejected`, and `metadata`.
- `prompt` is rendered with `render_prompt(example)`.
- `chosen` is the first strong solver attempt output.
- `rejected` is the first weak solver attempt output.
- `metadata` carries provenance from `example.metadata["judge"]`: `gap`, `weak_score`, `strong_score`, `quality`, and `tags` when present.
- Examples missing `weak_attempts` or `strong_attempts`, examples with empty attempt lists, and examples where chosen and rejected text are identical are skipped.
- Skips are counted and surfaced in the CLI summary line; they are not silently dropped.
- `--conversational` with `--format dpo` emits TRL conversational preference records where `prompt`, `chosen`, and `rejected` are role/content arrays.
- `--conversational` is rejected for non-DPO formats with a clear error.
- The public SDK export API has one entrypoint: `export_examples(...) -> ExportResult`.
- Serializer skip handling cannot write a partial record before counting a skip.
- Whitespace-only solver attempt outputs are treated as missing solver attempts.

## Unit Tests

- `tests/test_export.py` covers the `dpo` registry entry and unknown format error text including `dpo`.
- `tests/test_export.py` covers standard DPO serialization from examples with weak and strong attempts.
- `tests/test_export.py` covers conversational DPO serialization.
- `tests/test_export.py` covers skip cases: missing attempts, empty attempts, identical chosen/rejected output.
- `tests/test_export.py` covers DPO metadata provenance.
- `tests/test_export.py` covers `export_examples(...)` returning an `ExportResult`.
- `tests/test_export.py` covers whitespace-only solver attempt skips.

## Integration / Functional Tests

- CLI tests cover `datasmith export --format dpo --to local` from a run output directory.
- CLI tests cover the summary line including exported and skipped counts.
- CLI tests cover `--conversational` for DPO and rejection for non-DPO formats.

## Smoke Tests

- Run `.venv/bin/python -m pytest`.
- Run `.venv/bin/python -m ruff check .`.
- Run a local `datasmith export --format dpo` command against a temporary accepted JSONL and inspect the output shape.

## E2E Tests

- N/A — this is a local CLI export preset with no external service dependency.

## Manual / cURL Tests

- N/A — no HTTP API is changed.
