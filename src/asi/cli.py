from __future__ import annotations

import argparse
import json
from pathlib import Path

from asi.core import AgenticSelfInstruct
from asi.export import export_destinations, export_examples, export_formats
from asi.io import (
    export_run_result,
    export_seed_construction_result,
    read_jsonl,
    write_jsonl,
)
from asi.models import (
    DeterministicChallenger,
    DeterministicJudge,
    DeterministicSearchClient,
    DeterministicSeedConstructor,
    DeterministicSeedJudge,
    DeterministicSolver,
)
from asi.otel import examples_from_otlp, examples_from_span_jsonl
from asi.seed_constructor import SeedConstructor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datasmith", description="DataSmith SDK CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Write a starter config")
    init_parser.add_argument("--output", required=True)

    ingest_parser = subparsers.add_parser("ingest-otel", help="Convert OTEL traces/spans to seed JSONL")
    ingest_parser.add_argument("input")
    ingest_parser.add_argument("--output", required=True)
    ingest_parser.add_argument(
        "--format",
        choices=["otlp", "jsonl"],
        default="otlp",
        help="Input format: OTLP JSON export or flattened span JSONL",
    )

    construct_parser = subparsers.add_parser(
        "construct-seeds",
        help="Construct initial seed examples from a domain brief and search signals",
    )
    construct_parser.add_argument("--domain", required=True)
    construct_parser.add_argument("--output-dir", required=True)
    construct_parser.add_argument("--target-count", type=int, default=5)
    construct_parser.add_argument("--query", action="append", default=[])
    construct_parser.add_argument("--local-demo", action="store_true", help="Use deterministic local agents")

    run_parser = subparsers.add_parser("run", help="Run the weak-vs-strong data generation loop")
    run_parser.add_argument("--seeds", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--target-count", type=int, default=10)
    run_parser.add_argument("--weak-rollouts", type=int, default=3)
    run_parser.add_argument("--strong-rollouts", type=int, default=3)
    run_parser.add_argument("--local-demo", action="store_true", help="Use deterministic local models")

    export_parser = subparsers.add_parser(
        "export",
        help="Export trainer-ready datasets from accepted examples",
    )
    export_parser.add_argument("--from", dest="from_path", required=True)
    export_parser.add_argument(
        "--format",
        default="raw",
        help=f"Export format: {', '.join(export_formats())}",
    )
    export_parser.add_argument(
        "--to",
        dest="destination",
        default="local",
        help=f"Export destination: {', '.join(export_destinations())}",
    )
    export_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "init":
        _write_starter_config(Path(args.output))
        return 0
    if args.command == "ingest-otel":
        examples = (
            examples_from_otlp(args.input)
            if args.format == "otlp"
            else examples_from_span_jsonl(args.input)
        )
        write_jsonl(args.output, examples)
        print(f"wrote {len(examples)} seed examples to {args.output}")
        return 0
    if args.command == "construct-seeds":
        if not args.local_demo:
            raise SystemExit(
                "--local-demo is required until search/model provider config is supplied"
            )
        constructor = SeedConstructor(
            search_client=DeterministicSearchClient(),
            constructor=DeterministicSeedConstructor(),
            judge=DeterministicSeedJudge(),
        )
        result = constructor.run(
            domain=args.domain,
            target_count=args.target_count,
            queries=args.query or None,
        )
        export_seed_construction_result(result, args.output_dir)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if not args.local_demo:
            raise SystemExit("--local-demo is required by the CLI until provider config is supplied")
        seeds = read_jsonl(args.seeds)
        runner = AgenticSelfInstruct(
            challenger=DeterministicChallenger(),
            weak_solver=DeterministicSolver("weak"),
            strong_solver=DeterministicSolver("strong"),
            judge=DeterministicJudge(),
            weak_rollouts=args.weak_rollouts,
            strong_rollouts=args.strong_rollouts,
        )
        result = runner.run(seeds, target_count=args.target_count)
        export_run_result(result, args.output_dir)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        examples = read_jsonl(_export_input_path(args.from_path))
        try:
            record_count = export_examples(
                examples,
                format_name=args.format,
                destination_name=args.destination,
                output=args.output,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"exported {record_count} records to {args.output}")
        return 0
    return 1


def _export_input_path(source: str | Path) -> Path:
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"export source does not exist: {path}")
    if path.is_dir():
        accepted = path / "accepted.jsonl"
        if not accepted.exists():
            raise SystemExit(f"export source directory is missing accepted.jsonl: {path}")
        return accepted
    return path


def _write_starter_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "target_count: 100",
                "weak_rollouts: 3",
                "strong_rollouts: 3",
                "acceptance:",
                "  min_gap: 0.2",
                "  max_weak_score: 0.5",
                "  min_strong_score: 0.65",
                "otel:",
                "  prompt_keys: [gen_ai.prompt, llm.prompt, input.value]",
                "  completion_keys: [gen_ai.completion, llm.completion, output.value]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote starter config to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
