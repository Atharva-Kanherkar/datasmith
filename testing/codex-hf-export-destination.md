# codex/hf-export-destination — Test Contract

## Functional Behavior

- `datasmith export --from <dir> --format dpo --to hf --repo <user/dataset>` reads accepted examples and uploads the serialized records to a Hugging Face dataset repo.
- `--to hf` supports every registered export format because it consumes serialized records after format selection.
- `--repo` is required for `--to hf`.
- `--private` is passed through to dataset repo creation.
- Authentication uses `HF_TOKEN`.
- Missing `HF_TOKEN` fails with a clear CLI error.
- Missing Hugging Face optional dependencies fails with `pip install datasmith[hf]`.
- The destination uploads `data/train.jsonl`.
- The destination uploads a generated `README.md` dataset card.
- The dataset card includes DataSmith provenance: export format, record count, source, run summary fields when available, and an acceptance-policy note.

## Unit Tests

- `tests/test_export.py` covers the `hf` destination registry entry.
- `tests/test_export.py` mocks the Hugging Face API client and asserts repo creation arguments.
- `tests/test_export.py` asserts the uploaded JSONL payload.
- `tests/test_export.py` asserts the generated dataset card contents.
- `tests/test_export.py` covers missing repo, missing token, and missing optional dependency errors.

## Integration / Functional Tests

- CLI tests cover `datasmith export --to hf --repo ... --private` with a mocked Hub client.
- CLI tests cover run-directory summary ingestion for the dataset card.

## Smoke Tests

- Run `.venv/bin/python -m pytest`.
- Run `.venv/bin/python -m ruff check .`.

## E2E Tests

- Live Hub upload is intentionally not run in CI. Use a real `HF_TOKEN` manually:
  `datasmith export --from runs/demo --format dpo --to hf --repo <user/dataset> --private`.

## Manual / cURL Tests

- N/A — this is a CLI/SDK integration.
