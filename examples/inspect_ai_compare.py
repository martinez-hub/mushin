# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Is that eval gap real? Statistics for `Inspect AI <https://inspect.aisi.org.uk>`_ logs.

You ran the same eval on two models. One scored 60%, the other 56.7%. **Should
you switch?**

Inspect AI gives you those headline numbers — it runs the eval, the solvers, the
tool use, the scoring — but it does not tell you whether a gap is real. A gap can
be luck in two different ways, and this script measures both:

1. **Sampling luck.** Models generate different answers run to run. Re-run the
   same eval and the score wiggles. Inspect's ``--epochs`` gives you the repeats;
   mushin turns them into a p-value.
2. **Question luck.** You picked 50 questions. A *different* 50 might rank the
   models the other way round. This is usually the larger risk and the one
   nobody measures — a bootstrap over the eval items answers it.

A difference worth acting on has to survive both.

**The mushin part, in full.** Everything else in this file is the Inspect
adapter (:func:`scores_from_logs`) and the printing — worth reading if you are
writing your own adapter, but not what the library asks of you::

    from mushin.llm import compare_scores

    result = compare_scores(scores)          # {model: (n_epochs, n_items) array}
    row = result.comparisons.iloc[0]

    row["p_corrected"]                       # would a RE-RUN agree?
    row["item_p_corrected"]                  # would OTHER QUESTIONS agree?
    row["item_ci_low"], row["item_ci_high"]  # by how much, with an interval

Pass ``clusters=`` too when the questions are grouped (several per passage), or
the interval comes out too narrow.

Try it without installing anything or running a real eval::

    python examples/inspect_ai_compare.py --demo

That runs two scenarios with **known ground truth** — two identical models, and
one genuinely better model — so you can see the checks correctly refuse the
first and accept the second.

On your own logs::

    inspect eval theory_of_mind.py --model openai/gpt-4 --epochs 5
    inspect eval theory_of_mind.py --model anthropic/claude-3-5-sonnet --epochs 5
    python examples/inspect_ai_compare.py logs/*.eval

Requires ``pip install "mushin-py[eval]"``; the log-reading path additionally
needs ``pip install inspect-ai``.

How the two tools line up:

===================  ===============================================
Inspect AI           mushin
===================  ===============================================
one sample           one **item**  — "would other questions agree?"
one epoch            one **run**   — "would a re-run agree?"
===================  ===============================================
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

# Inspect's CORRECT/INCORRECT sentinels, as they appear in a log.
_VERDICTS = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _to_float(value: Any) -> float:
    """One Inspect ``Score.value`` -> float.

    Score values are not always numeric: the built-in scorers record ``"C"``/
    ``"I"`` for correct/incorrect. Anything unrecognised raises rather than being
    coerced — a silently wrong number here is invisible downstream.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        key = value.strip().upper()
        if key in _VERDICTS:
            return _VERDICTS[key]
        try:
            return float(key)
        except ValueError as exc:
            raise ValueError(
                f"cannot score Inspect value {value!r}; pass a `value_fn` that "
                "maps your scorer's values to floats"
            ) from exc
    raise TypeError(
        f"unsupported Inspect score value type {type(value).__name__} ({value!r}); "
        "pass a `value_fn` to convert it"
    )


def scores_from_logs(
    logs: Iterable[Any],
    *,
    scorer: str | None = None,
    value_fn=_to_float,
) -> tuple[dict[str, np.ndarray], list[Any]]:
    """Inspect ``EvalLog`` objects -> ``({model: (n_runs, n_items)}, question_ids)``.

    Questions are matched across models **by ``sample.id``, never by position**.
    Two Inspect logs can list their samples in different orders — retries,
    parallelism, a shuffled dataset — and the comparison pairs question *i* of one
    model with question *i* of the other. Matching positionally would compare
    "capital of France" against "solve this integral" and report a confident,
    meaningless answer. Every model must cover the same question ids, or this
    raises.

    ``scorer`` picks one entry from ``sample.scores`` when the task has several.
    """
    per_model: dict[str, dict[tuple[Any, int], float]] = {}
    ids_seen: dict[str, set[Any]] = {}
    tasks: dict[str, str] = {}

    for log in logs:
        model = log.eval.model
        task = getattr(log.eval, "task", None)
        if task is not None:
            tasks[model] = str(task)
        if model in per_model:
            raise ValueError(
                f"two logs for model {model!r}; pass one log per model (or rename "
                "them) so each column of the comparison is one system"
            )
        cells: dict[tuple[Any, int], float] = {}
        samples = log.samples or []
        if not samples:
            raise ValueError(
                f"log for {model!r} has no samples; it was probably read with "
                "header_only=True — use read_eval_log(...) instead"
            )
        for sample in samples:
            if sample.scores is None:
                raise ValueError(
                    f"sample {sample.id!r} in the log for {model!r} has no scores "
                    "(an errored or unscored Inspect sample). Re-run or filter "
                    "those samples out before comparing."
                )
            scores = sample.scores
            if scorer is None:
                if len(scores) != 1:
                    raise ValueError(
                        f"sample {sample.id!r} has scorers {sorted(scores)}; pass "
                        "scorer=<name> to choose one"
                    )
                (score,) = scores.values()
            else:
                if scorer not in scores:
                    raise ValueError(
                        f"scorer {scorer!r} missing from sample {sample.id!r} "
                        f"(has {sorted(scores)})"
                    )
                score = scores[scorer]
            epoch = getattr(sample, "epoch", 1)
            key = (sample.id, epoch)
            if key in cells:
                raise ValueError(
                    f"duplicate (id={sample.id!r}, epoch={epoch}) in the log for "
                    f"{model!r}"
                )
            cells[key] = float(value_fn(score.value))
        per_model[model] = cells
        ids_seen[model] = {sid for sid, _ in cells}

    models = list(per_model)
    # Inspect numbers samples 1..N *within a task*, so two logs from DIFFERENT
    # tasks have colliding ids that the id-based pairing below would happily
    # match: question 7 of one eval against question 7 of an unrelated one. The
    # ids alone cannot detect that, so check the task name before trusting them.
    if len(set(tasks.values())) > 1:
        listing = ", ".join(f"{m}={tasks[m]!r}" for m in sorted(tasks))
        raise ValueError(
            f"logs come from different Inspect tasks ({listing}), so their sample "
            "ids refer to different questions and cannot be paired. Compare models "
            "on one task at a time."
        )

    shared = ids_seen[models[0]]
    for model in models[1:]:
        if ids_seen[model] != shared:
            missing = sorted(map(str, shared - ids_seen[model]))[:5]
            extra = sorted(map(str, ids_seen[model] - shared))[:5]
            raise ValueError(
                f"models answered different questions, so they cannot be paired: "
                f"{model!r} is missing {missing} and adds {extra}. Run every model "
                "on the same dataset."
            )

    # Deterministic and order-independent, but sort ids in their OWN order where
    # they have one. Inspect numbers samples 1, 2, 3... by default, and sorting
    # those as strings gives 1, 10, 11, ... 2, 20 — internally consistent, so the
    # pairing stays correct, but this list is what a caller lines `clusters=` up
    # against, and a surprising order there misaligns the groups silently.
    try:
        question_ids = sorted(shared)
    except TypeError:  # mixed id types have no total order; fall back to str
        question_ids = sorted(shared, key=str)
    arrays: dict[str, np.ndarray] = {}
    for model, cells in per_model.items():
        epochs = sorted({ep for _, ep in cells})
        try:
            arrays[model] = np.array(
                [[cells[(sid, ep)] for sid in question_ids] for ep in epochs],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(
                f"log for {model!r} is missing a sample/epoch cell {exc.args[0]!r}; "
                "every sample must be scored in every epoch"
            ) from exc
    return arrays, question_ids


def _units(scores: dict[str, np.ndarray]):
    """Choose the display units, and return ``(level, signed, noun)`` formatters.

    Inspect's built-in scorers are 0-1, so percentages read naturally — but a
    custom scorer can return anything (a 1-10 rubric, a token count, a negative
    log-likelihood), and formatting 7.442 as ``744.2%`` is nonsense. Percentages
    are used only when every score actually lies in [0, 1].
    """
    finite = np.concatenate(
        [np.asarray(a, dtype=float).ravel() for a in scores.values()]
    )
    finite = finite[np.isfinite(finite)]
    fractional = bool(finite.size) and finite.min() >= 0.0 and finite.max() <= 1.0
    if fractional:
        return (lambda v: f"{v:6.1%}", lambda v: f"{v * 100:+.1f}", "points")
    return (lambda v: f"{v:6.3f}", lambda v: f"{v:+.3f}", "score units")


def _seed_verdict(row, n_runs, a, b):
    """The re-run axis -> ``(ok, sentence)``, where ok is None if unmeasurable.

    Uses the CORRECTED p-value: with three or more logs the raw one over-states
    significance, and the library already did the correction.
    """
    p = row["p_corrected"]
    if np.isnan(p):
        why = (
            "only one epoch, so re-run variation is unmeasured"
            if min(n_runs[a], n_runs[b]) < 2
            else "one model scored identically in every epoch, so it has no "
            "re-run distribution to test"
        )
        return None, f"no answer — {why}"
    ok = bool(row["significant"])
    return ok, (
        f"survives re-running (p={p:.4f})"
        if ok
        else f"could be re-run noise (p={p:.4f})"
    )


def _item_verdict(row, scores, a, b, gap, alpha, signed, noun):
    """The eval-set axis -> ``(ok, sentence)``, where ok is None if unmeasurable.

    Corrected over the same family as the seed axis: comparing three models
    multiplies the chance of a spurious item-level win exactly as it does on the
    seed axis.
    """
    lo, hi = row["item_ci_low"], row["item_ci_high"]
    if gap < 0:  # the interval is signed method_a - method_b; orient it to the
        lo, hi = -hi, -lo  # lead just announced, or it reads as a denial of it
    p = row["item_p_corrected"]
    if np.isnan(p):
        # compare_scores always has per-question scores, so the only way the
        # bootstrap declines to answer is a degenerate eval set.
        per_item_gap = scores[a].mean(axis=0) - scores[b].mean(axis=0)
        why = (
            "every question shows the same gap, so there is no eval-set "
            "variation to resample"
            if np.allclose(per_item_gap, per_item_gap.flat[0])
            else "the item bootstrap could not be computed"
        )
        return None, f"no answer — {why}"
    ok = p < alpha
    interval = f"95% CI [{signed(lo)}, {signed(hi)}] {noun}"
    return ok, (
        f"holds up (p={p:.4f}, {interval})"
        if ok
        else f"NOT established (p={p:.4f}, {interval} includes 0 — "
        "another question set could flip it)"
    )


def _outcome(seed_ok, item_ok) -> str:
    """Three outcomes, not two: a check that could not be RUN is not a check that
    FAILED. A temperature-0 model has no re-run distribution, and calling its
    large, item-significant gap "not defensible" would be wrong."""
    if seed_ok is False or item_ok is False:
        return "not a difference you can defend"
    if seed_ok and item_ok:
        return "a difference worth acting on"
    unmeasured = "re-run" if seed_ok is None else "other-questions"
    return (
        f"UNPROVEN — the {unmeasured} check could not be run, so only one of the "
        "two risks was ruled out"
    )


def report(
    scores: dict[str, np.ndarray], *, clusters=None, alpha: float = 0.05
) -> None:
    """Print the headline numbers, then whether each gap survives both checks.

    ``clusters`` is positional against the item axis, which is the sorted
    question-id order :func:`scores_from_logs` returns — pass its second return
    value through your grouping lookup, not the raw log order.
    """
    from mushin.llm import compare_scores

    names = list(scores)
    if len(names) < 2:
        print(
            f"only one model ({names[0]}) — a comparison needs at least two logs."
            if names
            else "no models to compare."
        )
        return

    level, signed, noun = _units(scores)
    n_runs = {name: arr.shape[0] for name, arr in scores.items()}
    # Recomputed from the per-sample scores, so it is the mean over the epochs
    # and samples present in the log — not necessarily the figure Inspect's own
    # aggregate metric printed (a different reducer would give a different one).
    print("Mean score per model (recomputed from the per-sample scores):")
    for name in names:
        print(f"   {name:<32} {level(scores[name].mean())}  ({n_runs[name]} epoch(s))")

    result = compare_scores(scores, clusters=clusters, alpha=alpha)
    n_pairs = len(result.comparisons)
    if n_pairs > 1:
        print(
            f"\n{n_pairs} pairwise comparisons — BOTH p-values below are "
            "Holm-corrected for multiplicity (the confidence intervals are "
            "per-comparison, not simultaneous)."
        )

    for _, row in result.comparisons.iterrows():
        a, b = row["method_a"], row["method_b"]
        gap = row["mean_diff"]
        leader, trailer = (a, b) if gap >= 0 else (b, a)
        print(
            f"\n{leader} leads {trailer} by {signed(abs(gap)).lstrip('+')} {noun}. "
            "Is that real?"
        )

        seed_ok, verdict = _seed_verdict(row, n_runs, a, b)
        print(f"   would a RE-RUN agree?          {verdict}")

        item_ok, items = _item_verdict(row, scores, a, b, gap, alpha, signed, noun)
        print(f"   would OTHER QUESTIONS agree?   {items}")

        print(f"   -> {_outcome(seed_ok, item_ok)}")


def _fake_log(model: str, question_skill, bonus: float, rng, epochs: int = 5):
    """A stand-in for one `inspect eval --epochs N` run, log-shaped."""
    rows = []
    for epoch in range(1, epochs + 1):
        for qid, skill in question_skill.items():
            p = 1.0 / (1.0 + np.exp(-(skill + bonus + rng.normal(scale=0.4))))
            rows.append((qid, epoch, "C" if rng.random() < p else "I"))
    rng.shuffle(rows)  # real logs are not ordered; alignment is by id
    samples = [
        types.SimpleNamespace(
            id=qid, epoch=ep, scores={"accuracy": types.SimpleNamespace(value=v)}
        )
        for qid, ep, v in rows
    ]
    return types.SimpleNamespace(
        eval=types.SimpleNamespace(model=model), samples=samples
    )


def demo() -> None:
    """Two scenarios with known ground truth, so the verdicts can be checked."""
    for title, bonus, truth in [
        ("SCENARIO 1 — the two models are IDENTICAL", 0.0, "any gap here is luck"),
        ("SCENARIO 2 — one model is GENUINELY BETTER", 1.6, "the gap is real"),
    ]:
        rng = np.random.default_rng(2)
        skill = {f"q{i}": rng.normal() for i in range(50)}
        logs = [
            _fake_log("gpt-4", skill, bonus, rng),
            _fake_log("claude-3-5", skill, 0.0, rng),
        ]
        scores, _ = scores_from_logs(logs)
        print("=" * 72)
        print(f"{title}  ({truth})")
        print("=" * 72)
        report(scores)
        print()


def main(argv: Sequence[str]) -> int:
    if "--demo" in argv:
        demo()
        return 0
    if not argv:
        print(__doc__)
        return 1
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print("reading real logs needs Inspect AI:  pip install inspect-ai")
        print("or try:  python examples/inspect_ai_compare.py --demo")
        return 1
    scores, question_ids = scores_from_logs([read_eval_log(p) for p in argv])
    print(f"{len(question_ids)} questions, aligned by sample id\n")
    report(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
