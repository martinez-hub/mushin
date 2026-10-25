# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Pin documented statistical thresholds to the code that enforces them.

The statistics guide tells a user who hit the underpowered-test warning how many
seeds will clear it. That number lives in prose, while the behaviour lives in
``warn_if_underpowered``, so the two can drift apart silently — and did: the
guide said 5 seeds made Wilcoxon viable when the floor is 6, so following the
documented remedy produced the same warning it promised to fix.

A paired Wilcoxon's best-case two-sided p-value is ``2 ** (1 - n)``: 0.25 at 3
seeds, 0.0625 at 5, 0.0313 at 6. Only the last clears ``alpha=0.05``.
"""

from __future__ import annotations

import functools
import re
import warnings
from pathlib import Path

import pytest

from _separated import separated, underpowered_warnings
from mushin.benchmark import compare_methods
from mushin.benchmark._stats import warn_if_underpowered

STATISTICS_GUIDE = (
    Path(__file__).resolve().parent.parent / "docs" / "guides" / "statistics.md"
)
ALPHA = 0.05


def _warns(test: str, n_seeds: int, alpha: float = ALPHA) -> bool:
    return bool(
        underpowered_warnings(warn_if_underpowered, test, n_seeds=n_seeds, alpha=alpha)
    )


def _seed_floor(test: str, alpha: float = ALPHA) -> int:
    """Smallest seed count at which `test` stops being flagged underpowered."""
    for n in range(2, 64):
        if not _warns(test, n, alpha):
            return n
    raise AssertionError(f"{test} never clears alpha={alpha}")


def test_wilcoxon_floor_is_six_seeds_not_five():
    """The arithmetic the guide now quotes: 2 ** (1 - n) first clears 0.05 at 6."""
    assert _warns("wilcoxon", 3) and _warns("wilcoxon", 5)
    assert not _warns("wilcoxon", 6)
    assert _seed_floor("wilcoxon") == 6


def test_guide_quotes_the_real_wilcoxon_seed_floor():
    """The number in the guide must be the number the warning enforces.

    If this fails, either the prose or `warn_if_underpowered` moved. Re-derive
    the floor and update the guide — do not just bump the constant here.
    """
    text = STATISTICS_GUIDE.read_text(encoding="utf-8")
    match = re.search(r"first clears `alpha=0\.05` at \*\*(\d+)\*\* seeds", text)
    assert match, (
        "could not find the documented Wilcoxon seed floor in "
        f"{STATISTICS_GUIDE.name}; if the wording changed, update this test"
    )
    documented = int(match.group(1))
    assert documented == _seed_floor("wilcoxon"), (
        f"{STATISTICS_GUIDE.name} says {documented} seeds make Wilcoxon viable, "
        f"but warn_if_underpowered still warns below {_seed_floor('wilcoxon')}"
    )
    assert _warns("wilcoxon", documented - 1), (
        "the documented floor is not tight: one seed fewer should still warn"
    )


def test_guide_does_not_recommend_the_default_test_for_small_n():
    """Wilcoxon is `compare`'s default and is the worst choice at small n.

    The test-selection table used to recommend it for "small n" while the same
    page documented that it cannot reach alpha there.
    """
    row = next(
        line
        for line in STATISTICS_GUIDE.read_text(encoding="utf-8").splitlines()
        if line.startswith('| `"wilcoxon"`')
    )
    assert "small n" not in row.lower(), (
        "the statistics guide recommends the default Wilcoxon test for small n, "
        "which it cannot serve: it needs "
        f"{_seed_floor('wilcoxon')} seeds to reach alpha={ALPHA}"
    )


# Every test's floor, not just Wilcoxon's. The LLM guide once grouped
# `ttest_rel` with the rank tests and said the group "cannot" reach alpha at
# small n. `ttest_rel` is paired but parametric and its floor is 2, so that was
# false: only the RANK tests have a floor above the minimum. Pinning all five
# keeps any future claim about any of them honest.
EXPECTED_FLOORS = {
    "wilcoxon": 6,
    "mannwhitney": 4,
    "welch": 2,
    "ttest_rel": 2,
    "ttest_ind": 2,
}


@pytest.mark.parametrize("test,expected", sorted(EXPECTED_FLOORS.items()))
def test_seed_floor_per_test(test: str, expected: int):
    assert _seed_floor(test) == expected, (
        f"{test}'s seed floor moved; the guides quote these numbers"
    )


@pytest.mark.parametrize("test", ["welch", "ttest_rel", "ttest_ind"])
def test_parametric_tests_can_reach_alpha_at_three_seeds(test: str):
    """The guide's other remedy — switch to a parametric test — must hold.

    `ttest_rel` is included deliberately: it is *paired*, and the guide used to
    imply the paired tests were all underpowered at small n.
    """
    assert not _warns(test, 3)


def test_guides_do_not_call_a_parametric_test_underpowered():
    """No "cannot reach alpha" sentence may name a test whose floor is 2.

    This is the claim that was wrong: a sentence saying the "rank/paired tests
    ... cannot" reach alpha, naming `ttest_rel` alongside `wilcoxon`.
    """
    parametric = [t for t, floor in EXPECTED_FLOORS.items() if floor <= 2]
    # The original defect elided the verb: "Welch ... can at least reach `alpha`
    # at 3 seeds, whereas the rank/paired tests (`wilcoxon`, `ttest_rel`) cannot
    # at small _n_". So match the negation plus any underpowered-ness marker,
    # not "cannot reach" specifically.
    negation = re.compile(r"\bcan(?:not|'t)\b|\bunable to\b")
    marker = re.compile(
        r"`?alpha`?|p *[<=] *0?\.\d|small _?n_?|underpowered|too few seeds"
    )
    for guide in ("llm.md", "statistics.md"):
        text = (STATISTICS_GUIDE.parent / guide).read_text(encoding="utf-8")
        for sentence in re.split(r"(?<=[.!?])\s", text):
            flat = " ".join(sentence.split())
            hit = negation.search(flat)
            if not (hit and marker.search(flat)):
                continue
            # Only the SUBJECT of the negation matters. "the parametric tests
            # (`welch`, `ttest_rel`) can reach alpha, while the rank tests
            # cannot" is correct and must not trip this.
            before = flat[: hit.start()]
            boundary = max(
                (
                    m.end()
                    for m in re.finditer(
                        r"[,;:]|\b(?:while|whereas|but|though|although)\b", before
                    )
                ),
                default=0,
            )
            subject = before[boundary:]
            named = [t for t in parametric if f"`{t}`" in subject]
            assert not named, (
                f"docs/guides/{guide} implies {named} cannot reach alpha at "
                f"small n, but their floor is 2 seeds: {flat[:170]}"
            )


AGENT_GUIDE = (
    Path(__file__).resolve().parent.parent / "src" / "mushin" / "agent_guide.md"
)


@functools.cache  # a brute-force search, asked for the same three counts repeatedly
def _holm_floor(n_methods: int, alpha: float = ALPHA) -> int:
    """Seeds at which compare_methods first calls every pair significant on data
    that separates the methods perfectly."""
    for n in range(2, 32):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the underpowered warning, below the floor
            if compare_methods(separated(n_methods, n), alpha=alpha)[
                "significant"
            ].all():
                return n
    raise AssertionError(f"{n_methods} methods never clear alpha={alpha}")


def _compare_warns(n_methods: int, n_seeds: int, alpha: float = ALPHA) -> bool:
    return bool(
        underpowered_warnings(
            compare_methods, separated(n_methods, n_seeds), alpha=alpha
        )
    )


@pytest.mark.parametrize("n_methods, floor", [(2, 6), (3, 7), (4, 8)])
def test_holm_seed_floor(n_methods: int, floor: int):
    """With m pairs, Holm needs m * 2 ** (1 - n) < alpha."""
    assert _holm_floor(n_methods) == floor


@pytest.mark.parametrize("n_methods", [2, 3, 4])
def test_underpowered_warning_fires_exactly_below_the_holm_floor(n_methods: int):
    """The warning promises that no pair can be significant. It must fire one
    seed below the floor at which every pair is, and not at the floor, or
    following the guides' seed counts would keep the warning (or lose it while
    nothing can be significant)."""
    floor = _holm_floor(n_methods)
    assert _compare_warns(n_methods, floor - 1), f"silent below the floor ({floor})"
    assert not _compare_warns(n_methods, floor), f"warns at the floor ({floor})"


@pytest.mark.parametrize("guide", [AGENT_GUIDE, STATISTICS_GUIDE])
def test_guides_quote_the_holm_seed_floors(guide: Path):
    text = guide.read_text(encoding="utf-8")
    match = re.search(
        r"at least (\d+) seeds for 2\s+methods, (\d+) for 3 and (\d+)\s+for 4", text
    )
    assert match, f"could not find the seed floors in {guide.name}"
    assert tuple(map(int, match.groups())) == tuple(_holm_floor(k) for k in (2, 3, 4))
