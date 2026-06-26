"""compare_llms: run systems across seeds, score, and compare with significance."""

from __future__ import annotations

import inspect
import json
import os
import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from torchmetrics import Metric as TorchMetric

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import (
    available_tests,
    compare_methods,
    holm_correction,
)

from ._cache import OutputCache
from ._system import as_system
from ._types import Metric


def _normalize_examples(data: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    inputs, refs = [], []
    for ex in data:
        if isinstance(ex, dict) and "input" in ex:
            inputs.append(ex["input"])
            refs.append(ex.get("reference"))
        elif isinstance(ex, tuple) and len(ex) == 2:
            inputs.append(ex[0])
            refs.append(ex[1])
        else:
            inputs.append(ex)
            refs.append(None)
    return inputs, refs


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _accepts_seed(m) -> bool:
    """True if the callable metric takes a `seed` argument (e.g. an llm_judge
    metric), so the per-trial seed can be threaded through to it."""
    try:
        return "seed" in inspect.signature(m).parameters
    except (TypeError, ValueError):
        return False


def _score_one(
    name: str | None, m: Metric, outputs, refs, seed: int
) -> dict[str, float]:
    """Score a batch with one metric -> {data_var_name: value} (dicts expand)."""
    if isinstance(m, TorchMetric):
        m.reset()
        m.update(outputs, refs)  # user shapes data per the metric's update signature
        value = m.compute()
        base = name if name is not None else _snake(type(m).__name__)
        if isinstance(value, dict):
            return {
                (f"{base}_{k}" if name is not None else str(k)): float(v)
                for k, v in value.items()
            }
        return {base: float(value)}
    # plain callable: mean of per-example scores. Pass the trial seed if the metric
    # accepts one (e.g. llm_judge), so a stochastic judge is tied to the run.
    base = name if name is not None else "score"
    pass_seed = _accepts_seed(m)
    scores = []
    for i, (o, r) in enumerate(zip(outputs, refs)):
        try:
            scores.append(float(m(o, r, seed=seed) if pass_seed else m(o, r)))
        except Exception as e:
            raise type(e)(f"metric {base!r} failed on example {i}: {e}") from e
    return {base: sum(scores) / len(scores)}


def _is_constant(values) -> bool:
    """True if seed-to-seed values have no meaningful within-group variance.

    Uses ``np.allclose`` rather than exact equality so sub-epsilon float jitter —
    e.g. from the non-associative ``sum(scores) / len(scores)`` reduction — counts
    as constant and is masked, instead of leaking into a catastrophic-cancellation
    "significant" p-value when ``compare_methods`` runs the parametric test on it.
    This mirrors the ``np.allclose`` short-circuit ``compare_methods`` already uses
    between systems."""
    arr = np.asarray(values, dtype=float)
    return bool(np.allclose(arr, arr[0]))


def _score(metrics, outputs, refs, seed: int) -> dict[str, float]:
    row: dict[str, float] = {}
    if isinstance(metrics, dict):
        for name, m in metrics.items():
            row.update(_score_one(name, m, outputs, refs, seed))
    else:
        row.update(_score_one(None, metrics, outputs, refs, seed))
    return row


def _normalize_output(out: Any) -> Any:
    """Round-trip an output through JSON (tuple -> list, int keys -> str keys) so a
    score does not depend on whether `cache=` was supplied. Cached replays are
    always JSON (the cache stores JSON), so we apply the same normalization to
    fresh outputs on *both* the cached and uncached paths. Best-effort: an output
    that is not JSON-serializable (and so could not be cached anyway) passes
    through unchanged, keeping the no-cache path usable for non-JSON outputs."""
    try:
        return json.loads(json.dumps(out))
    except (TypeError, ValueError):
        return out


def _run(system, inputs, seed, cache, name) -> list[Any]:
    if cache is None:
        return [_normalize_output(out) for out in system(inputs, seed)]
    cached, missing = cache.partition(name, seed, inputs)
    if missing:
        # Call the system on ONLY the missing inputs. This assumes output[i]
        # depends only on input[i] and the seed (not on batch composition) — the
        # usual one-prompt-one-completion case. Documented in the guide.
        fresh = list(system([inp for _, inp in missing], seed))
        if len(fresh) != len(missing):
            raise ValueError(
                f"system {name!r} seed {seed} returned {len(fresh)} outputs for "
                f"{len(missing)} inputs"
            )
        cache.put_many(
            name, seed, [(inp, out) for (_, inp), out in zip(missing, fresh)]
        )
        # Normalize fresh outputs the same way (see _normalize_output) so a fresh
        # run scores exactly what a later cached replay — and a no-cache run —
        # would score.
        for (i, _), out in zip(missing, fresh):
            cached[i] = _normalize_output(out)
    return [cached[i] for i in range(len(inputs))]


def compare_llms(
    systems: dict[str, Any],
    data: Sequence[Any],
    metric: Metric | dict[str, Metric],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    test: str = "welch",
    alpha: float = 0.05,
    cache: str | os.PathLike[str] | None = None,
) -> BenchmarkResult:
    if not systems:
        raise ValueError("`systems` is empty")
    if test not in available_tests():
        # Validate up front so a typo'd test name fails before any (possibly
        # token-spending) system calls or cache writes.
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    if isinstance(metric, dict) and not metric:
        raise ValueError("`metric` battery is empty; provide at least one metric")
    inputs, refs = _normalize_examples(data)
    if not inputs:
        raise ValueError("`data` is empty")
    # Validate seeds up front — before instantiating systems, which for hydra-zen
    # configs can load large models or trigger provider setup. Duplicate seeds are
    # the same (system, seed) trial: counting them as independent samples would
    # understate variance and inflate significance, so reject them.
    seeds = list(seeds)
    if not seeds:
        raise ValueError("`seeds` is empty; provide at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"`seeds` contains duplicates ({seeds}); each seed is one trial and "
            "must be unique — repeats are not independent samples."
        )

    sysmap = {name: as_system(v) for name, v in systems.items()}
    store = OutputCache(cache) if cache is not None else None

    results: dict[str, list[dict[str, float]]] = {}
    for name, system in sysmap.items():
        per_seed = []
        for seed in seeds:
            outputs = _run(system, inputs, seed, store, name)
            if len(outputs) != len(inputs):
                raise ValueError(
                    f"system {name!r} seed {seed} returned {len(outputs)} outputs "
                    f"for {len(inputs)} inputs"
                )
            per_seed.append(_score(metric, outputs, refs, seed))
        results[name] = per_seed

    # Detect zero within-group variance per (metric, system): a metric whose
    # scores are identical across all seeds has no sampling distribution. Warn
    # only for systems that are constant in *every* metric (they ignore the seed).
    zero_var: dict[str, set[str]] = {}  # metric -> systems with zero variance
    if len(seeds) > 1:
        for name, per_seed in results.items():
            keys = list(per_seed[0])
            constant = [k for k in keys if _is_constant([row[k] for row in per_seed])]
            for k in constant:
                zero_var.setdefault(k, set()).add(name)
            if len(constant) == len(keys):
                warnings.warn(
                    f"system {name!r} produced identical scores across all "
                    f"{len(seeds)} seeds — it likely ignores the seed or is "
                    "deterministic, so seed-based significance involving it is "
                    "not meaningful (the seeds are duplicated points, not "
                    "independent samples). Wire the seed to sampling, or treat "
                    "its score as a point estimate.",
                    UserWarning,
                    stacklevel=2,
                )

    ds = to_dataset(results)
    ds = ds.assign_coords(seed=list(seeds))  # use the actual seed values, not 0..n-1
    comparisons = compare_methods(ds, test=test, alpha=alpha)

    if zero_var:
        # A (metric, system) with zero variance has no valid sampling distribution,
        # so its comparisons are not real significance tests — mask them per metric
        # rather than report a duplicated-point p-value of ~0.
        def _involves_zero_var(row) -> bool:
            zv = zero_var.get(row["metric"], ())
            return row["method_a"] in zv or row["method_b"] in zv

        mask = comparisons.apply(_involves_zero_var, axis=1)
        # NaN the effect_size too: a standardized effect divided by ~zero
        # within-group variance is a meaningless ±inf/huge artifact, and reporting
        # it next to significant=False is contradictory. mean_diff is left intact —
        # it is a valid descriptive statistic regardless of significance.
        comparisons.loc[mask, ["p_value", "p_corrected", "effect_size"]] = float("nan")
        comparisons.loc[mask, "significant"] = False

        # Re-apply the Holm correction per metric over only the *surviving*
        # comparisons, so they are not over-corrected for the excluded
        # zero-variance pairs (which compare_methods had counted in the family).
        for _, group in comparisons.groupby("metric"):
            valid = group.index[~mask.loc[group.index]]
            if len(valid):
                pvals = comparisons.loc[valid, "p_value"].tolist()
                corrected = holm_correction(pvals) if len(pvals) > 1 else pvals
                comparisons.loc[valid, "p_corrected"] = [float(c) for c in corrected]
                comparisons.loc[valid, "significant"] = [
                    False if c != c else bool(c < alpha) for c in corrected
                ]

    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
