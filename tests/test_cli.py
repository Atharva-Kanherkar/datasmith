from __future__ import annotations

import json

from asi.cli import main


def test_cli_ingest_otel_writes_jsonl(tmp_path) -> None:
    otlp = tmp_path / "traces.json"
    output = tmp_path / "seeds.jsonl"
    otlp.write_text(
        json.dumps(
            {
                "resourceSpans": [
                    {
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "traceId": "t",
                                        "spanId": "s",
                                        "name": "chat",
                                        "attributes": [
                                            {
                                                "key": "gen_ai.prompt",
                                                "value": {"stringValue": "Prompt"},
                                            }
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["ingest-otel", str(otlp), "--output", str(output)]) == 0
    assert "Prompt" in output.read_text(encoding="utf-8")


def test_cli_run_local_demo_writes_outputs(tmp_path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    output_dir = tmp_path / "run"
    seeds.write_text('{"input":{"task":"x"},"expected":{"answer":"y"}}\n', encoding="utf-8")

    assert (
        main(
            [
                "run",
                "--seeds",
                str(seeds),
                "--output-dir",
                str(output_dir),
                "--target-count",
                "1",
                "--local-demo",
            ]
        )
        == 0
    )
    assert (output_dir / "accepted.jsonl").exists()
    assert (output_dir / "summary.json").exists()


def test_cli_construct_seeds_local_demo_writes_seed_outputs(tmp_path) -> None:
    output_dir = tmp_path / "seed-run"

    assert (
        main(
            [
                "construct-seeds",
                "--domain",
                "legal refund policy reasoning",
                "--output-dir",
                str(output_dir),
                "--target-count",
                "1",
                "--local-demo",
            ]
        )
        == 0
    )
    assert (output_dir / "seeds.jsonl").exists()
    assert (output_dir / "rejected-seeds.jsonl").exists()
    assert (output_dir / "signals.json").exists()
    assert (output_dir / "summary.json").exists()
