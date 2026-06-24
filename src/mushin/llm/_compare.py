"""compare_llms: run systems across seeds, score, and compare with significance."""

from __future__ import annotations

import inspect
import os
import re
import warnings
from collections.abc import Sequence
from typing import Any

from torchmetrics import Metric as TorchMetric

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import compare_methods

from ._cache import OutputCache
from ._system import as_system
from ._types import Metric


def _normalize_examples(data: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    inputs, refs = [], []
    for ex in data:
        if isinstance(ex, dict):
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


def _score(metrics, outputs, refs, seed: int) -> dict[str, float]:
    row: dict[str, float] = {}
    if isinstance(metrics, dict):
        for name, m in metrics.items():
            row.update(_score_one(name, m, outputs, refs, seed))
    else:
        row.update(_score_one(None, metrics, outputs, refs, seed))
    return row


def _run(system, inputs, seed, cache, name) -> list[Any]:
    if cache is None:
        return list(system(inputs, seed))
    cached, missing = cache.partition(name, seed, inputs)
    if missing:
        fresh = list(system([inp for _, inp in missing], seed))
        if len(fresh) != len(missing):
            raise ValueError(
                f"system {name!r} seed {seed} returned {len(fresh)} outputs for "
                f"{len(missing)} inputs"
            )
        cache.put_many(
            name, seed, [(inp, out) for (_, inp), out in zip(missing, fresh)]
        )
        for (i, _), out in zip(missing, fresh):
            cached[i] = out
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
    inputs, refs = _normalize_examples(data)
    if not inputs:
        raise ValueError("`data` is empty")

    sysmap = {name: as_system(v) for name, v in systems.items()}
    store = OutputCache(cache) if cache is not None else None
    seeds = list(seeds)

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

    deterministic: set[str] = set()
    if len(seeds) > 1:
        for name, per_seed in results.items():
            if all(len({row[k] for row in per_seed}) == 1 for k in per_seed[0]):
                deterministic.add(name)
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

    if deterministic:
        # A system with zero within-group variance has no valid sampling
        # distribution, so any comparison involving it is not a real significance
        # test — force it not-significant rather than report a duplicated-point
        # p-value of ~0.
        mask = comparisons["method_a"].isin(deterministic) | comparisons[
            "method_b"
        ].isin(deterministic)
        comparisons.loc[mask, "significant"] = False
        for col in ("p_value", "p_corrected"):
            if col in comparisons.columns:
                comparisons.loc[mask, col] = float("nan")

    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
