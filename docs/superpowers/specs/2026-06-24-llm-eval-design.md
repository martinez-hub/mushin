# LLM evaluation — Design

*Date: 2026-06-24*

## Goal

Extend mushin's reproducible, statistically-rigorous comparison from torch
classifiers to **LLM systems** (models / prompts / configs). Provide
`compare_llms`: run each system across reproducible stochastic seeds, score with
a user-supplied metric, and compare with significance — reusing the existing
`(method × seed) → Holm-corrected significance → BenchmarkResult` machinery. The
differentiator: almost no LLM eval reports whether a difference is real vs
sampling noise; mushin's `compare` is built for exactly that.

This is **Piece B** of the LLM/agentic direction. Agentic multi-step eval is a
separate later effort (Piece C).

## Positioning (owned vs delegated)

mushin owns the **evaluate + report + statistics** spine and nothing else. It
**delegates**: the systems (the user's callables wrapping any provider/local
model), the eval data, the metric, the prompts, rate-limiting/concurrency, and
all provider/API specifics. mushin ships **no** datasets, **no** provider
adapters, **no** model IDs — it stays provider-agnostic the same way it delegates
training in the torch path.

## Architecture

New submodule `src/mushin/llm/`, **reusing** the framework-agnostic statistics
and reporting from `mushin.benchmark`:

- `compare_llms` builds a `(method × seed)` `xarray.Dataset` of scalar metrics
  and calls the existing `mushin.benchmark._stats.compare_methods` →
  `BenchmarkResult`. It does **not** touch the torch `evaluate`/task-registry
  (different contract: systems not `nn.Module`, plain-callable metrics not
  torchmetrics, no device/no_grad).
- The torch `compare` and `compare_llms` share only the stats + reporting layer
  (`_stats`, `_aggregate.to_dataset`, `_result.BenchmarkResult`).

Files:
- `src/mushin/llm/__init__.py` — exports `compare_llms`, `llm_judge`, the
  `System`/`Metric` type aliases, and `Example`.
- `src/mushin/llm/_compare.py` — `compare_llms` (orchestration: run → score →
  aggregate → compare).
- `src/mushin/llm/_judge.py` — the `llm_judge` helper.
- `src/mushin/llm/_cache.py` — the on-disk output cache.

## Core API

```python
def compare_llms(
    systems: dict[str, System],
    data: Sequence[Example],
    metric: Metric | dict[str, Metric],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    test: str = "welch",
    alpha: float = 0.05,
    cache: str | os.PathLike[str] | None = None,
    progress: bool = True,
) -> BenchmarkResult: ...
```

### Contracts

- `System = Callable[[Sequence[Any], int], Sequence[Any]]` — given the list of
  example **inputs** and a **seed**, returns a list of **outputs** (same length,
  same order). The system wires the seed to its own sampling (provider `seed`
  param, local RNG, temperature) for reproducibility. Batching/concurrency/
  rate-limits are the system's responsibility.
- `Example` — a mapping with at least `"input"` (passed to the system) and,
  when the metric needs it, `"reference"` (the gold answer). Extra keys are
  ignored. A plain `(input, reference)` tuple is also accepted and normalized.
- `Metric = Callable[[output, reference], float]` — scores one example. mushin
  aggregates the per-example scores into one value per `(system, seed)` by mean.
  A `dict[str, Metric]` defines a battery (each becomes a data variable, exactly
  like the torch battery). A metric that ignores `reference` (e.g. a
  reference-free judge) is fine.

### Behavior

`test`, `alpha`, and the returned `BenchmarkResult` (`.data`, `.comparisons`,
`.summary()`) are identical to the torch path — same significance, same Holm
correction, same underpowered-test warning, same single-seed/zero-variance
handling. A `seeds` of length 1, or a deterministic system that yields identical
scores across seeds, produces zero within-system variance and is reported as
**not** significant (no false positives) — the existing `_stats` logic, reused
unchanged.

## The `llm_judge` helper

```python
def llm_judge(
    judge: Callable[[str, int], str],
    rubric: str,
    *,
    parse: Callable[[str], float] = parse_score,
    template: Callable[[str, Any, Any], str] = default_template,
) -> Metric: ...
```

Turns a **user-supplied, provider-agnostic** judge-call function into a pointwise
`Metric`. For each example it builds a judge prompt from `rubric` + the system
`output` + the `reference` (via `template`), calls `judge(prompt, seed)` (seeded
for reproducibility — the same seed mushin used for that run), and parses the
judge's reply to a float in `[0, 1]` via `parse` (default extracts a leading
0/1, yes/no, or `score: X`). mushin owns the prompt-format/seed/parse plumbing;
the user owns the judge model. No provider code, no model IDs.

## Caching (in v1)

An optional on-disk cache (`cache=<dir>`) keyed by
`(system_name, seed, sha256(input))` storing the system's `output`. For each
`(system, seed)` mushin partitions the inputs into cached vs missing, calls
`system(missing_inputs, seed)` on **only the missing ones**, writes those
outputs through to the cache, and merges cached + fresh outputs back into the
original example order before scoring. (So the system must map any subset of
inputs → outputs in order — which the batch contract already guarantees.) This
makes reruns, resumes, and partial-failure recovery free — essential given LLM
cost — and reinforces reproducibility (a fully-cached run replays exactly with
no calls).

- Storage: one JSON-lines file per `(system, seed)` under the cache dir, or a
  small SQLite db (decided in the plan after a quick check; JSONL is the default
  for zero-dependency simplicity).
- The cache stores only system **outputs**, not metric scores (metrics are cheap
  and may change); judge calls inside `llm_judge` may be cached under the same
  scheme keyed by the judge prompt (decided in the plan).
- `cache=None` (default) disables it.

## Concurrency (delegated, v1)

mushin calls `system(inputs, seed)` with the **whole** input list and runs seeds
sequentially. The system owns batching, async, and rate-limiting (it knows its
provider). mushin manages no event loop, thread pool, or rate limiter in v1.

## Data flow

For each `system × seed`:
1. (cache-aware) obtain `outputs = system(inputs, seed)`.
2. score each `(output, reference)` with each metric → per-example scores.
3. mean over examples → one scalar per metric → one `(system, seed)` row.
Assemble all rows into the `(method × seed)` `xarray.Dataset` (via
`_aggregate.to_dataset`) → `compare_methods(test, alpha)` → `BenchmarkResult`.

## Error handling

- A `System` returning the wrong number of outputs → `ValueError` naming the
  system/seed and the length mismatch.
- A metric raising on an example → propagate with the example index and system
  in the message (don't silently drop).
- Unknown `test` → the existing `NotImplementedError` from `_stats`.
- Empty `systems`/`data` → `ValueError`.
- A judge reply `parse` can't interpret → `ValueError` showing the raw reply, so
  the user can fix their `parse`/`rubric`.

## Testing strategy

All hermetic — **no network, no real LLM**:

- **Fake systems:** a deterministic fake (returns a fixed mapping → zero variance
  across seeds) and a stochastic fake (uses `seed` to perturb outputs → real
  variance). Assert `compare_llms` returns a `BenchmarkResult` with dims
  `(method, seed)`, the metric present, and significance behaving correctly
  (clear winner flagged; deterministic-tie not flagged).
- **Metric battery:** a dict of two metrics → two data variables.
- **`llm_judge`:** a fake judge function (deterministic, seed-aware) + `parse` →
  verify it scores and is reproducible; a malformed judge reply → `ValueError`.
- **Caching:** run twice with `cache=tmp_path`; assert the system is called the
  first time and **not** the second (use a call-counter fake); assert results are
  identical; assert a partial cache (some examples present) only calls the
  missing ones.
- **Reproducibility:** same seeds + same (seed-respecting) fake → identical
  `result.data`.
- **Statistics reuse:** confirm `result.comparisons` columns/behavior match the
  torch path (it's the same `compare_methods`).

## Non-goals (v1)

- Pairwise / A-vs-B judging (separate aggregation + significance path) — later.
- Bundled datasets, provider adapters, or model IDs — delegated permanently.
- Agentic / multi-step / tool-use evaluation — **Piece C**, separate spec.
- mushin-managed concurrency, async, or rate-limiting — delegated to the system.
- A `Study`-like train+compare orchestration for LLMs (there's no training) —
  out of scope; `compare_llms` is the unit.
