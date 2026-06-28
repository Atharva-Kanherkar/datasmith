from __future__ import annotations

import json

from asi.io import export_run_result, read_jsonl, write_jsonl
from asi.types import Example, RunResult


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [Example(input={"q": "x"}, expected={"a": "y"}, metadata={"m": 1})])

    examples = read_jsonl(path)

    assert examples == [Example(input={"q": "x"}, expected={"a": "y"}, metadata={"m": 1})]


def test_export_run_result_writes_artifacts(tmp_path) -> None:
    result = RunResult(accepted=[Example(input={"q": "x"})], attempts=1)

    export_run_result(result, tmp_path)

    assert (tmp_path / "accepted.jsonl").exists()
    assert (tmp_path / "rejected.jsonl").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 1
