from __future__ import annotations

import json

import pytest

from asi.cli import main
from asi.export import (
    export_destinations,
    export_examples,
    export_formats,
    get_destination,
    get_format,
    render_prompt,
)
from asi.types import Example


def test_export_registries_expose_raw_and_local() -> None:
    assert export_formats() == ["raw"]
    assert export_destinations() == ["local"]
    assert get_format("raw")
    assert get_destination("local")


def test_unknown_export_format_lists_valid_choices() -> None:
    with pytest.raises(ValueError, match="unknown export format: nope. Valid formats: raw"):
        get_format("nope")


def test_unknown_export_destination_lists_valid_choices() -> None:
    with pytest.raises(
        ValueError,
        match="unknown export destination: nowhere. Valid destinations: local",
    ):
        get_destination("nowhere")


def test_render_prompt_uses_string_input_as_is() -> None:
    assert render_prompt(Example(input="Write the answer")) == "Write the answer"


def test_render_prompt_uses_first_known_dict_key() -> None:
    example = Example(input={"llm.prompt": "Trace prompt", "gen_ai.prompt": "Preferred prompt"})

    assert render_prompt(example) == "Preferred prompt"


def test_render_prompt_accepts_custom_prompt_keys() -> None:
    example = Example(input={"custom.prompt": "Custom prompt"})

    assert render_prompt(example, prompt_keys=("custom.prompt",)) == "Custom prompt"


def test_render_prompt_compacts_dict_without_known_keys() -> None:
    example = Example(input={"task": "classify", "labels": ["yes", "no"]})

    assert render_prompt(example) == '{"labels":["yes","no"],"task":"classify"}'


def test_raw_local_export_writes_example_dicts(tmp_path) -> None:
    output = tmp_path / "dataset.jsonl"
    examples = [Example(input={"q": "x"}, expected={"a": "y"}, metadata={"m": 1})]

    count = export_examples(examples, format_name="raw", destination_name="local", output=output)

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "expected": {"a": "y"},
        "input": {"q": "x"},
        "metadata": {"m": 1},
    }


def test_cli_export_reads_run_directory_accepted_jsonl(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "accepted.jsonl").write_text(
        '{"input":{"q":"x"},"expected":{"a":"y"},"metadata":{"m":1}}\n',
        encoding="utf-8",
    )
    output = tmp_path / "dataset.jsonl"

    assert (
        main(
            [
                "export",
                "--from",
                str(run_dir),
                "--format",
                "raw",
                "--to",
                "local",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert output.read_text(encoding="utf-8") == (
        '{"expected": {"a": "y"}, "input": {"q": "x"}, "metadata": {"m": 1}}\n'
    )


def test_cli_export_reads_direct_jsonl_path(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text('{"input":"Prompt","metadata":{}}\n', encoding="utf-8")
    output = tmp_path / "dataset.jsonl"

    assert (
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--format",
                "raw",
                "--to",
                "local",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert output.read_text(encoding="utf-8") == '{"input": "Prompt", "metadata": {}}\n'


def test_cli_export_unknown_format_exits_with_valid_choices(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text('{"input":"Prompt","metadata":{}}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--format",
                "nope",
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )

    assert str(exc.value) == "unknown export format: nope. Valid formats: raw"


def test_cli_export_unknown_destination_exits_with_valid_choices(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text('{"input":"Prompt","metadata":{}}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--to",
                "nowhere",
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )

    assert str(exc.value) == "unknown export destination: nowhere. Valid destinations: local"
