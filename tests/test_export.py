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
    render_messages,
    render_prompt,
)
from asi.types import Example


def dpo_example(
    *,
    weak: str = "Weak answer",
    strong: str = "Strong answer",
    metadata: dict | None = None,
) -> Example:
    return Example(
        input={"prompt": "Explain the refund decision"},
        metadata={
            "weak_attempts": [{"role": "weak", "output": weak, "attempt": 1, "metadata": {}}],
            "strong_attempts": [
                {"role": "strong", "output": strong, "attempt": 1, "metadata": {}}
            ],
            "judge": {
                "gap": 0.4,
                "weak_score": 0.2,
                "strong_score": 0.6,
                "quality": "high",
                "tags": ["refunds"],
                "reason": "extra field omitted",
            },
            **(metadata or {}),
        },
    )


def test_export_registries_expose_raw_and_local() -> None:
    assert export_formats() == ["dpo", "messages", "prompt_completion", "raw"]
    assert export_destinations() == ["local"]
    assert get_format("dpo")
    assert get_format("raw")
    assert get_format("prompt_completion")
    assert get_format("messages")
    assert get_destination("local")


def test_unknown_export_format_lists_valid_choices() -> None:
    with pytest.raises(
        ValueError,
        match="unknown export format: nope. Valid formats: dpo, messages, prompt_completion, raw",
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


def test_render_messages_preserves_existing_message_array() -> None:
    messages = [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": "Can I refund this account?"},
    ]

    assert render_messages(Example(input={"gen_ai.input.messages": messages})) == messages
    assert render_messages(Example(input=messages)) == messages


def test_render_completion_stringifies_expected_output() -> None:
    assert render_completion(Example(input="Prompt", expected={"answer": "Yes"})) == '{"answer":"Yes"}'


def test_raw_local_export_writes_example_dicts(tmp_path) -> None:
    output = tmp_path / "dataset.jsonl"
    examples = [Example(input={"q": "x"}, expected={"a": "y"}, metadata={"m": 1})]

    result = export_examples(examples, format_name="raw", destination_name="local", output=output)

    assert result.records == 1
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

    result = export_examples(
        examples,
        format_name="prompt_completion",
        destination_name="local",
        output=output,
    )

    assert result.records == 1
    assert result.skipped == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "prompt": "Decide refund",
        "completion": '{"answer":"Deny automatic refund"}',
    }


def test_messages_export_writes_chat_records_without_metadata(tmp_path) -> None:
    output = tmp_path / "messages.jsonl"

    result = export_examples(
        [Example(input="Question", expected="Answer", metadata={"judge": {"gap": 0.5}})],
        format_name="messages",
        destination_name="local",
        output=output,
    )

    assert result.records == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }


def test_dpo_export_writes_preference_records_with_metadata(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"

    result = export_examples(
        [dpo_example()],
        format_name="dpo",
        destination_name="local",
        output=output,
    )

    assert result.records == 1
    assert result.skipped == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "prompt": "Explain the refund decision",
        "chosen": "Strong answer",
        "rejected": "Weak answer",
        "metadata": {
            "gap": 0.4,
            "weak_score": 0.2,
            "strong_score": 0.6,
            "quality": "high",
            "tags": ["refunds"],
        },
    }


def test_dpo_export_uses_first_solver_attempts(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"
    example = dpo_example()
    example.metadata["weak_attempts"].append(
        {"role": "weak", "output": "Second weak answer", "attempt": 2, "metadata": {}}
    )
    example.metadata["strong_attempts"].append(
        {"role": "strong", "output": "Second strong answer", "attempt": 2, "metadata": {}}
    )

    result = export_examples([example], format_name="dpo", destination_name="local", output=output)

    assert result.records == 1
    assert json.loads(output.read_text(encoding="utf-8"))["chosen"] == "Strong answer"
    assert json.loads(output.read_text(encoding="utf-8"))["rejected"] == "Weak answer"


def test_dpo_conversational_export_writes_message_arrays(tmp_path) -> None:
    output = tmp_path / "dpo-messages.jsonl"

    result = export_examples(
        [dpo_example()],
        format_name="dpo",
        destination_name="local",
        output=output,
        conversational=True,
    )

    assert result.records == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["prompt"] == [{"role": "user", "content": "Explain the refund decision"}]
    assert record["chosen"] == [{"role": "assistant", "content": "Strong answer"}]
    assert record["rejected"] == [{"role": "assistant", "content": "Weak answer"}]


def test_dpo_conversational_export_preserves_prompt_messages(tmp_path) -> None:
    output = tmp_path / "dpo-messages.jsonl"
    messages = [
        {"role": "system", "content": "Use the policy."},
        {"role": "user", "content": "Can this account be refunded?"},
    ]
    example = dpo_example()
    example.input = {"gen_ai.input.messages": messages}

    result = export_examples(
        [example],
        format_name="dpo",
        destination_name="local",
        output=output,
        conversational=True,
    )

    assert result.records == 1
    assert json.loads(output.read_text(encoding="utf-8"))["prompt"] == messages


def test_dpo_export_skips_missing_solver_attempts(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"

    result = export_examples(
        [Example(input="Prompt", metadata={})],
        format_name="dpo",
        destination_name="local",
        output=output,
    )

    assert result.records == 0
    assert result.skipped == 1
    assert result.skip_reasons == {"no solver attempts": 1}
    assert output.read_text(encoding="utf-8") == ""


def test_dpo_export_skips_empty_solver_attempts(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"

    result = export_examples(
        [dpo_example(metadata={"weak_attempts": []})],
        format_name="dpo",
        destination_name="local",
        output=output,
    )

    assert result.records == 0
    assert result.skipped == 1
    assert result.skip_reasons == {"no solver attempts": 1}


def test_dpo_export_skips_whitespace_only_solver_output(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"

    result = export_examples(
        [dpo_example(weak="   ")],
        format_name="dpo",
        destination_name="local",
        output=output,
    )

    assert result.records == 0
    assert result.skipped == 1
    assert result.skip_reasons == {"no solver attempts": 1}


def test_dpo_export_skips_identical_chosen_and_rejected(tmp_path) -> None:
    output = tmp_path / "dpo.jsonl"

    result = export_examples(
        [dpo_example(weak="Same answer", strong="Same answer")],
        format_name="dpo",
        destination_name="local",
        output=output,
    )

    assert result.records == 0
    assert result.skipped == 1
    assert result.skip_reasons == {"identical chosen/rejected": 1}


def test_conversational_flag_requires_dpo_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="--conversational is only supported with format dpo"):
        export_examples(
            [Example(input="Question", expected="Answer")],
            format_name="messages",
            destination_name="local",
            output=tmp_path / "out.jsonl",
            conversational=True,
        )


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
        "unknown export format: nope. Valid formats: dpo, messages, prompt_completion, raw"
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


def test_cli_export_dpo_reports_skipped_examples(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "accepted.jsonl").write_text(
        "\n".join(
            [
                json.dumps(dpo_example().to_dict()),
                json.dumps(Example(input="Prompt", metadata={}).to_dict()),
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dpo.jsonl"

    assert (
        main(
            [
                "export",
                "--from",
                str(run_dir),
                "--format",
                "dpo",
                "--to",
                "local",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert json.loads(output.read_text(encoding="utf-8"))["chosen"] == "Strong answer"
    assert capsys.readouterr().out.strip() == (
        f"exported 1 records to {output} (skipped 1: 1 no solver attempts)"
    )


def test_cli_export_dpo_conversational(tmp_path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(json.dumps(dpo_example().to_dict()) + "\n", encoding="utf-8")
    output = tmp_path / "dpo.jsonl"

    assert (
        main(
            [
                "export",
                "--from",
                str(accepted),
                "--format",
                "dpo",
                "--conversational",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(record["prompt"], list)
    assert isinstance(record["chosen"], list)
    assert isinstance(record["rejected"], list)
