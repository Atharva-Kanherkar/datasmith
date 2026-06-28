from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from asi.seed_constructor import SeedConstructionResult
from asi.types import Example, RunResult


def read_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            examples.append(Example.from_dict(raw))
    return examples


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def export_run_result(result: RunResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "accepted.jsonl", result.accepted)
    with (output / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for item in result.rejected:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_seed_construction_result(result: SeedConstructionResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "seeds.jsonl", result.accepted)
    with (output / "rejected-seeds.jsonl").open("w", encoding="utf-8") as handle:
        for item in result.rejected:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    (output / "signals.json").write_text(
        json.dumps(
            [signal.to_dict() for signal in result.signals],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
