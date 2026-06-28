from __future__ import annotations

from asi import (
    DeterministicSearchClient,
    DeterministicSeedConstructor,
    DeterministicSeedJudge,
    SeedConstructor,
)


def test_seed_constructor_uses_search_signals_and_accepts_seeds() -> None:
    constructor = SeedConstructor(
        search_client=DeterministicSearchClient(),
        constructor=DeterministicSeedConstructor(),
        judge=DeterministicSeedJudge(),
    )

    result = constructor.run(domain="legal refund policy reasoning", target_count=2)

    assert len(result.accepted) == 2
    assert result.summary()["target_met"] is True
    assert result.summary()["signals"] >= 1
    assert result.accepted[0].metadata["generator"] == "datasmith-seed-constructor"
    assert result.accepted[0].metadata["web_signals"]


def test_seed_constructor_rejects_invalid_target_count() -> None:
    constructor = SeedConstructor(
        search_client=DeterministicSearchClient(),
        constructor=DeterministicSeedConstructor(),
        judge=DeterministicSeedJudge(),
    )

    try:
        constructor.run(domain="legal", target_count=0)
    except ValueError as exc:
        assert "target_count" in str(exc)
    else:
        raise AssertionError("expected ValueError")
