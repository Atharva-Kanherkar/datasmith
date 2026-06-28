# Contributing

Thanks for helping improve Agentic Self-Instruct.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Principles

- Keep the SDK provider-agnostic.
- Do not require API keys for tests.
- Preserve trace metadata when ingesting OTEL-shaped data.
- Prefer explicit typed artifacts over ad hoc dictionaries at the public boundary.
- Document any behavior inspired by research papers without overstating reproduction claims.
