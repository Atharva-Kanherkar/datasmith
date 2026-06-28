from __future__ import annotations

import json

from asi.types import Example, SolverAttempt


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def challenger_prompt(seeds: list[Example], feedback: list[str], target_count: int) -> str:
    lines = [
        "You are the challenger in an Agentic Self-Instruct synthetic data loop.",
        "Create one challenging, realistic training example as strict JSON:",
        '{"input": {...}, "expected": {...}, "metadata": {"tags": []}}',
        "The example should be solvable by a strong model but likely difficult for a weaker model.",
        f"Target dataset size: {target_count}.",
        "",
        "Seed examples:",
    ]
    for index, seed in enumerate(seeds, start=1):
        lines.append(f"{index}. input: {compact_json(seed.input)}")
        if seed.expected is not None:
            lines.append(f"   expected: {compact_json(seed.expected)}")
    if feedback:
        lines.append("")
        lines.append("Recent judge feedback:")
        lines.extend(f"- {item}" for item in feedback[-8:])
    return "\n".join(lines)


def solver_prompt(role: str, candidate: Example) -> str:
    return "\n".join(
        [
            f"You are the {role} solver in an Agentic Self-Instruct loop.",
            "Solve the task. Return only your final answer.",
            "Do not mention dataset generation, judging, or hidden references.",
            "",
            "Candidate input:",
            compact_json(candidate.input),
        ]
    )


def judge_prompt(
    candidate: Example,
    weak_attempts: list[SolverAttempt],
    strong_attempts: list[SolverAttempt],
) -> str:
    lines = [
        "You are the judge in an Agentic Self-Instruct synthetic data loop.",
        "Score whether the candidate is high quality and whether strong solvers outperform weak solvers.",
        "Return only JSON with this shape:",
        '{"verdict":"accept|reject","weak_score":0.0,"strong_score":1.0,"gap":0.0,'
        '"quality":"high|medium|low","reason":"","feedback":"","tags":[]}',
        "",
        "Candidate input:",
        compact_json(candidate.input),
    ]
    if candidate.expected is not None:
        lines.extend(["Reference answer:", compact_json(candidate.expected)])
    lines.append("")
    lines.append("Weak attempts:")
    lines.extend(f"- {attempt.output}" for attempt in weak_attempts)
    lines.append("")
    lines.append("Strong attempts:")
    lines.extend(f"- {attempt.output}" for attempt in strong_attempts)
    return "\n".join(lines)
