"""compare_llms: run systems across seeds, score, and compare with significance."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from torchmetrics import Metric as TorchMetric

from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import (
    available_corrections,
    available_tests,
    compare_methods,
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
        else:
            # Anything else is a bare input (reference=None). We deliberately do
            # NOT treat a 2-tuple as (input, reference): a tuple-valued input such
            # as (system_prompt, user_prompt) would otherwise be silently split and
            # only its first element sent to the system. To attach a reference, use
            # the explicit {"input": ..., "reference": ...} mapping.
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


def _to_scalar(v) -> float:
    """Reduce a metric value to one float. A scalar passes through; a per-example
    sequence/tensor (e.g. `BERTScore` returns per-prediction precision/recall/f1)
    is averaged over examples — the same reduction applied to plain-callable
    scores — instead of raising on `float()` of a multi-element tensor."""
    if hasattr(v, "detach"):  # torch tensor -> CPU numpy
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


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
                (f"{base}_{k}" if name is not None else str(k)): _to_scalar(v)
                for k, v in value.items()
            }
        return {base: _to_scalar(value)}
    # plain callable: mean of per-example scores. Pass the trial seed if the metric
    # accepts one (e.g. llm_judge), so a stochastic judge is tied to the run.
    base = name if name is not None else "score"
    pass_seed = _accepts_seed(m)
    scores = []
    for i, (o, r) in enumerate(zip(outputs, refs, strict=True)):
        try:
            scores.append(float(m(o, r, seed=seed) if pass_seed else m(o, r)))
        except Exception as e:
            raise type(e)(f"metric {base!r} failed on example {i}: {e}") from e
    return {base: sum(scores) / len(scores)}


def _score(metrics, outputs, refs, seed: int) -> dict[str, float]:
    row: dict[str, float] = {}
    if isinstance(metrics, dict):
        for name, m in metrics.items():
            scored = _score_one(name, m, outputs, refs, seed)
            # A dict-returning metric expands to `<name>_<subkey>`, which can collide
            # with another battery entry (e.g. {"squad": SQuAD(), "squad_f1": ...}).
            # Silently overwriting would report the wrong metric, so reject it.
            clash = row.keys() & scored.keys()
            if clash:
                raise ValueError(
                    f"metric battery produced colliding output name(s) "
                    f"{sorted(clash)}; a dict-returning metric expands to "
                    "`<name>_<subkey>` — rename the battery key(s) to avoid the clash."
                )
            row.update(scored)
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
            name,
            seed,
            [(inp, out) for (_, inp), out in zip(missing, fresh, strict=True)],
        )
        # Normalize fresh outputs the same way (see _normalize_output) so a fresh
        # run scores exactly what a later cached replay — and a no-cache run —
        # would score.
        for (i, _), out in zip(missing, fresh, strict=True):
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
    correction: str = "holm",
    cache: str | os.PathLike[str] | None = None,
) -> BenchmarkResult:
    if not systems:
        raise ValueError("`systems` is empty")
    if test not in available_tests():
        # Validate up front so a typo'd test name fails before any (possibly
        # token-spending) system calls or cache writes.
        raise ValueError(f"unknown test {test!r}; choose from {available_tests()}")
    if correction not in available_corrections():
        raise ValueError(
            f"unknown correction {correction!r}; choose from {available_corrections()}"
        )
    if isinstance(metric, dict) and not metric:
        raise ValueError("`metric` battery is empty; provide at least one metric")
    inputs, refs = _normalize_examples(data)
    if not inputs:
        raise ValueError("`data` is empty")
    # Normalize references through the same JSON round-trip applied to outputs (see
    # _normalize_output), so output and reference are scored on equal footing — e.g.
    # a structured label like ("a", 1) matches an identical output, which would
    # otherwise become ["a", 1] and compare unequal.
    refs = [_normalize_output(r) for r in refs]
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

    ds = to_dataset(results)
    ds = ds.assign_coords(seed=list(seeds))  # use the actual seed values, not 0..n-1
    # compare_methods masks zero within-group-variance comparisons (a system
    # constant across seeds has no sampling distribution), re-Holms the survivors,
    # and warns for systems constant in every metric — so the LLM and torch paths
    # handle deterministic systems identically.
    comparisons = compare_methods(ds, test=test, alpha=alpha, correction=correction)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=alpha)
