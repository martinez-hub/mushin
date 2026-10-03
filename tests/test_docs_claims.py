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

import re
import warnings
from pathlib import Path

import pytest

from mushin.benchmark._stats import warn_if_underpowered

STATISTICS_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "guides" / "statistics.md"
ALPHA = 0.05


def _warns(test: str, n_seeds: int, alpha: float = ALPHA) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_underpowered(test, n_seeds=n_seeds, alpha=alpha)
    return any("cannot reach alpha" in str(c.message) for c in caught)


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


@pytest.mark.parametrize("test", ["welch", "ttest_ind"])
def test_parametric_tests_can_reach_alpha_at_three_seeds(test: str):
    """The guide's other remedy — switch to a parametric test — must hold."""
    assert not _warns(test, 3)
