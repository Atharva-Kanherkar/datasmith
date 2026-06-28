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
    render_completion,
    render_prompt,
)
from asi.types import Example


def test_export_registries_expose_raw_and_local() -> None:
    assert export_formats() == ["messages", "prompt_completion", "raw"]
    assert export_destinations() == ["local"]
    assert get_format("raw")
    assert get_format("prompt_completion")
    assert get_format("messages")
    assert get_destination("local")


def test_unknown_export_format_lists_valid_choices() -> None:
    with pytest.raises(
        ValueError,
        match="unknown export format: nope. Valid formats: messages, prompt_completion, raw",
    ):
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


def test_render_prompt_uses_otel_ingestion_prompt_keys() -> None:
    assert (
        render_prompt(Example(input={"openinference.input.value": "OpenInference prompt"}))
        == "OpenInference prompt"
    )
    assert render_prompt(Example(input={"prompt": "Plain prompt"})) == "Plain prompt"
    assert render_prompt(Example(input={"input": "Input prompt"})) == "Input prompt"


def test_render_prompt_accepts_custom_prompt_keys() -> None:
    example = Example(input={"custom.prompt": "Custom prompt"})

    assert render_prompt(example, prompt_keys=("custom.prompt",)) == "Custom prompt"


def test_render_prompt_compacts_dict_without_known_keys() -> None:
    example = Example(input={"task": "classify", "labels": ["yes", "no"]})

    assert render_prompt(example) == '{"labels":["yes","no"],"task":"classify"}'


def test_render_completion_stringifies_expected_output() -> None:
    assert render_completion(Example(input="Prompt", expected={"answer": "Yes"})) == '{"answer":"Yes"}'


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


def test_prompt_completion_export_writes_trainer_records_without_metadata(tmp_path) -> None:
    output = tmp_path / "dataset.jsonl"
    examples = [
        Example(
            input={"prompt": "Decide refund"},
            expected={"answer": "Deny automatic refund"},
            metadata={"judge": {"gap": 0.5}, "weak_attempts": ["leak"]},
        )
    ]

    count = export_examples(
        examples,
        format_name="prompt_completion",
        destination_name="local",
        output=output,
    )

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "prompt": "Decide refund",
        "completion": '{"answer":"Deny automatic refund"}',
    }


def test_messages_export_writes_chat_records_without_metadata(tmp_path) -> None:
    output = tmp_path / "messages.jsonl"

    count = export_examples(
        [Example(input="Question", expected="Answer", metadata={"judge": {"gap": 0.5}})],
        format_name="messages",
        destination_name="local",
        output=output,
    )

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }


def test_trainer_formats_require_expected_output(tmp_path) -> None:
    with pytest.raises(ValueError, match="prompt_completion export requires examples"):
        export_examples(
            [Example(input="Question")],
            format_name="prompt_completion",
            destination_name="local",
            output=tmp_path / "out.jsonl",
        )


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
                "prompt_completion",
                "--to",
                "local",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert output.read_text(encoding="utf-8") == '{"completion": "{\\"a\\":\\"y\\"}", "prompt": "{\\"q\\":\\"x\\"}"}\n'


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


def test_cli_export_defaults_to_prompt_completion(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text('{"input":"Prompt","expected":"Completion","metadata":{"judge":{}}}\n', encoding="utf-8")
    output = tmp_path / "dataset.jsonl"

    assert (
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert output.read_text(encoding="utf-8") == (
        '{"completion": "Completion", "prompt": "Prompt"}\n'
    )


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

    assert str(exc.value) == (
        "unknown export format: nope. Valid formats: messages, prompt_completion, raw"
    )


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


def test_cli_export_bad_jsonl_exits_cleanly(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text('{"metadata":{}}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--format",
                "prompt_completion",
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )

    assert str(exc.value) == "example is missing required field: input"
