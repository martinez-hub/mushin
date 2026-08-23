# LLM evaluation

!!! note "Requires the `eval` extra"
    LLM evaluation (`compare_llms`, `llm_judge`) is part of mushin's optional
    evaluation layer — install it with `pip install "mushin-py[eval]"`. See
    [Installation](../install.md#optional-extras).

The "vibes eval" problem: it's common to eyeball a handful of outputs and
declare a new prompt or model "better" — but that impression may be pure
sampling noise. `mushin.llm.compare_llms` runs each system across
reproducible stochastic seeds, scores every `(system, seed)` pair with a
user-supplied metric, and reports pairwise significance (Holm-corrected by
default; pass `correction=` for Bonferroni, Benjamini–Hochberg FDR, or none). The
same statistical spine that powers `mushin.benchmark.compare` for torch
models works here — you just bring the systems, data, and metric.

> **Prefer to follow along?** [Notebook 05 — LLM evaluation](../notebooks/05_llm_eval.ipynb)
> runs a full `compare_llms` example with outputs and a per-seed score plot.

## Quickstart

```python
--8 < --"examples/compare_llms_demo.py:run"
```

`systems` maps a name to a callable `system(inputs, seed) -> outputs`. The
seed is passed so each system can wire it to its own sampling parameter
(provider `seed` param, local RNG, temperature). mushin manages no concurrency
or rate limiting — that's the system's responsibility.

## Seeds and stochasticity

Each `(system, seed)` pair is one "trial". mushin runs each system once per
seed and computes a single aggregate score (mean over examples) for that trial.
The variance **across seeds** is what the significance test operates on.

A deterministic system (same output for every seed) produces zero
within-system variance. mushin reports this as **not** significant rather than
producing a false positive — exactly the same behavior as the torch path.

!!! danger "Seeds alone answer the *smaller* question"
    Seed variance is decoding/judge noise. It says nothing about the **eval set**,
    which is itself a sample — and "would this hold on a different 200 prompts?"
    is usually both the question you care about and the far larger uncertainty.
    A worked case in this repo's tests has a seed SD of `0.001` against an item
    SD of `1.0` (~1000x): the seed test returns `p = 5e-15` for a difference
    whose item-level interval straddles 0 (`item_p = 0.23`, i.e. ~12% of
    resampled eval sets reverse the winner).

    That is why `compare_llms` also runs a **paired item-level bootstrap** and
    puts it in the same table — see [Item-level uncertainty](#item-level-uncertainty).
    Read both. A seed-significant result whose item interval straddles 0 is not
    a result you should report.

```python
result = compare_llms(
    {"gpt4": gpt4_system, "claude": claude_system},
    data=eval_data,
    metric=exact_match,
    seeds=range(10),  # 10 seeds → more power
    test="welch",
)
```

Use more seeds (≥ 5) for a robust estimate. Welch's t-test (the default) already
has reasonable power at 3–5 seeds; the rank/paired tests (`wilcoxon`,
`ttest_rel`) are weak at small _n_ — a paired Wilcoxon over 3 seeds can never go
below p = 0.25. `compare_llms` warns when the test you chose cannot reach `alpha`
at the given seed count.

!!! warning "Paired tests need a *shared* per-seed random effect"
    `wilcoxon`/`ttest_rel` pair trial *k* of one system with trial *k* of the
    other. That pairing is only meaningful when seed *k* induces a shared
    random effect across systems (e.g. both score the same seed-*k* data
    subsample). For independent systems whose seed only drives their own
    sampling — the typical API-backed setup — the trials are uncorrelated and
    the pairing assumption does not hold; stick with the default `welch`.

## Item-level uncertainty

Alongside the seed-based test, `compare_llms` resamples the **eval items** with
replacement (a paired bootstrap, Koehn 2004) and reports four extra columns on
`result.comparisons`:

| column | meaning |
| --- | --- |
| `item_diff` | mean per-item difference (seed-averaged) |
| `item_ci_low` / `item_ci_high` | bootstrap CI on that difference |
| `item_p` | two-sided bootstrap p-value |

```python
row = result.comparisons.iloc[0]
row["p_value"]  # decoding noise only
row["item_ci_low"], row["item_ci_high"]  # would other prompts agree?
```

Two caveats worth knowing:

- **Only for per-item metrics.** Plain callables (including `llm_judge`) are
  scored per example, so they get item statistics. torchmetrics metrics are
  `update(batch)` → `compute()` and expose only an aggregate, so their item
  columns are `NaN` — scoring them per example would change what the metric
  means (corpus BLEU is not the mean of sentence BLEU).
- **It does not rescue a deterministic system.** Item variance exists at
  temperature 0, so the bootstrap still answers the eval-set question there,
  but the seed-based columns remain masked.

Pass `item_bootstrap=0` to skip it, or a different resample count (default
`10_000`).

### Grouped items need `clusters=`

Eval items are often **not independent** — several questions about one passage,
several prompts from one document. Those items share whatever makes their source
easy or hard, so resampling them individually treats correlated observations as
independent and reports an interval that is too narrow.

Pass a per-item group label and the bootstrap resamples **whole groups** instead:

```python
result = compare_llms(..., clusters=passage_id_per_item)
```

In this repo's tests (25 groups of 8, true difference 0), ignoring the grouping
covers a nominal 95% interval only about half the time; clustering restores it to
~93% and widens the interval roughly 3x. If your items are genuinely independent,
omit it — with one item per group the two paths are numerically identical.

!!! warning "Few clusters are still unreliable"
    The correction fixes the *unit* of resampling, not the sample size. With a
    handful of groups the interval is still miscalibrated — measured ~53%
    false-positive rate at 2 clusters and ~15% at 5, against a nominal 5%. What
    counts is the number of **groups**, not items: a 500-item eval set with 4
    passages is a 4-sample problem. `mushin` warns below ~20 groups. NaN group
    labels are rejected outright, because an unlabelled item would otherwise be
    dropped from every resample while still counting toward the point estimate.

## Bring your own scores

If another harness already produced per-item results — inspect-ai, lm-eval-harness,
your own runner — you do not have to hand mushin your execution loop to get the
analysis:

```python
from mushin.llm import compare_scores

result = compare_scores(
    {"gpt": gpt_per_item_scores, "claude": claude_per_item_scores},
    clusters=passage_id_per_item,  # optional
)
```

Each system maps to either a 1-D sequence of per-item scores (a single run — the
item-level columns are populated and the seed-based ones are masked, because one
run says nothing about decoding noise) or a 2-D `(n_seeds, n_items)` array, which
gives both dimensions. Systems must cover the same items in the same order; call
once per metric.

## Using another harness (Inspect AI)

You ran the same eval on two models with [Inspect AI](https://inspect.aisi.org.uk).
One scored 60%, the other 56.7%. **Should you switch?**

Inspect runs the evaluation and reports those headline numbers; it does not tell
you whether the gap is real. A gap can be luck in two ways, and mushin measures
both:

| the question | what could go wrong | where the evidence comes from |
| --- | --- | --- |
| Would a **re-run** agree? | models sample randomly, so the score wiggles run to run | Inspect's `--epochs` repeats |
| Would **other questions** agree? | you picked 50 questions; another 50 might rank them the other way | a bootstrap over the eval items |

A difference worth acting on has to survive both. The second is usually the
larger risk, and it is the one most harnesses never report.

See it work with no install and no eval run:

```bash
pip install "mushin-py[eval]"
python examples/inspect_ai_compare.py --demo
```

That runs two scenarios with **known ground truth** — two identical models, and
one genuinely better — so you can check the verdicts rather than trust them:

```
SCENARIO 1 — the two models are IDENTICAL  (any gap here is luck)
Mean score per model (recomputed from the per-sample scores):
   gpt-4        50.8%  (5 epoch(s))
   claude-3-5   46.0%  (5 epoch(s))

gpt-4 leads claude-3-5 by 4.8 points. Is that real?
   would a RE-RUN agree?          could be re-run noise (p=0.2484)
   would OTHER QUESTIONS agree?   NOT established (p=0.2560, 95% CI [-3.6, +13.2] points includes 0)
   -> not a difference you can defend
```

On your own logs:

```bash
inspect eval theory_of_mind.py --model openai/gpt-4 --epochs 5
inspect eval theory_of_mind.py --model anthropic/claude-3-5-sonnet --epochs 5
python examples/inspect_ai_compare.py logs/*.eval
```

The two tools line up directly: an Inspect **sample** is a mushin **item**, and
an Inspect **epoch** is a mushin **run** — so `--epochs 5` gives both dimensions.

!!! warning "Match questions by id, not position"
    The comparison pairs question *i* of one model with question *i* of the
    other. Two Inspect logs can list their samples in different orders — retries,
    parallelism, a shuffled dataset — so matching positionally would compare
    "capital of France" against "solve this integral" and report a confident,
    meaningless answer. `scores_from_logs` matches on `sample.id` and raises if
    the models did not answer the same questions. Do the same in any adapter you
    write.

[`examples/inspect_ai_compare.py`](https://github.com/martinez-hub/mushin/blob/main/examples/inspect_ai_compare.py)
carries the adapter (~70 lines to copy — mushin takes no Inspect AI dependency,
so it does not ship as an import). It also converts Inspect's `"C"`/`"I"`
verdicts. If your questions are grouped — several per passage — pass the group ids as
`clusters=` (see [Grouped items](#grouped-items-need-clusters)). It is positional
against the item axis, which is the **sorted question-id order** that
`scores_from_logs` returns as its second value — build the labels from that list,
not from the log order.

## Metric options

### Plain callable

A per-example scorer `(output, reference) -> float`. mushin means the
per-example scores into one value per `(system, seed)`:

```python
def exact_match(output, reference):
    return float(output.strip() == reference.strip())


result = compare_llms(systems, data, metric=exact_match, seeds=range(5))
```

### torchmetrics text metrics

Pass any `torchmetrics.Metric` object. mushin calls `metric.update(outputs,
references)` with the full batch then `metric.compute()`, resetting between
`(system, seed)` pairs. The error-rate metrics, which take flat string lists,
plug in directly:

```python
from torchmetrics.text import CharErrorRate, WordErrorRate

result = compare_llms(
    systems,
    data,  # each example's `reference` is a plain string
    metric={"wer": WordErrorRate(), "cer": CharErrorRate()},
    seeds=range(5),
)
```

A metric whose `compute()` returns a **dict** (e.g. `SQuAD` → `exact_match`,
`f1`) expands into one data variable per key. In a **named battery**
(`metric={"squad": SQuAD()}`) each key is prefixed with your name →
`squad_exact_match`, `squad_f1`; a **single bare metric** (`metric=SQuAD()`)
keeps the raw subkeys → `exact_match`, `f1`. A metric value that is a
**per-example sequence/tensor** (e.g. `BERTScore`'s per-prediction
`precision`/`recall`/`f1`) is averaged over the examples to a single trial score.

!!! warning "Shape `output`/`reference` to the metric"
    mushin passes your raw `output`s and `reference`s straight to
    `metric.update(outputs, references)` — it does **not** reshape them. Each
    torchmetrics text metric expects a specific shape, so shape your example
    `reference` (and `output`) accordingly:

    | metric | `output` | `reference` (per example) |
    |---|---|---|
    | `WordErrorRate`, `CharErrorRate`, `MatchErrorRate` | `str` | `str` |
    | `BLEUScore`, `SacreBLEUScore`, `CHRFScore` | `str` | **`list[str]`** (one or more references) |
    | `SQuAD` | `{"prediction_text": str, "id": str}` | `{"answers": {...}, "id": str}` |

    Passing a plain `str` reference to `BLEUScore` does **not** error but scores
    wrong; passing plain strings to `SQuAD` raises. If a metric doesn't fit this
    `(output, reference)` shape, wrap it in a plain `Callable[[output,
    reference], float]` instead.

!!! note "Extra deps"
    Some torchmetrics text metrics need optional packages. `WordErrorRate`,
    `CharErrorRate`, `BLEUScore`, and `SQuAD` work without extras. `ROUGEScore`
    needs `nltk`; install it separately (`uv add nltk`) and run
    `nltk.download("punkt")` before use.

### A battery of metrics

Pass a `dict[str, Metric]` to score with multiple metrics at once — each
becomes its own data variable in the result:

```python
result = compare_llms(
    systems,
    data,
    metric={"exact": exact_match, "wer": WordErrorRate()},
    seeds=range(5),
)
```

### `llm_judge`

Turn a judge LLM into a pointwise metric. You supply the judge callable
`judge(prompt, seed) -> reply` (wrapping any provider/local model); mushin
handles the prompt template, seed passing, and reply parsing:

```python
from mushin.llm import llm_judge


def my_judge(prompt, seed):
    # call your preferred provider here
    ...


metric = llm_judge(my_judge, rubric="Is this answer factually correct?")
result = compare_llms(systems, data, metric=metric, seeds=range(5))
```

See [API Reference — llm](../reference/llm.md) for the full signature.

## Hydra-zen system configs

Systems can be hydra-zen configs (`builds(...)` output) instead of raw
callables. mushin instantiates each system **once** before the seed loop, so a
heavy local model loads a single time and is reused across all seeds:

```python
from hydra_zen import builds
from mushin.llm import compare_llms

HFGeneratorConf = builds(
    MyHFGenerator,  # your class wrapping HF from_pretrained
    model_name="mistralai/Mistral-7B-v0.1",
    device_map="auto",  # shard across CPU/GPU automatically
    torch_dtype="float16",
)

result = compare_llms(
    systems={"mistral": HFGeneratorConf, "baseline": baseline_system},
    data=eval_data,
    metric=exact_match,
    seeds=range(5),
)
```

mushin delegates the actual device placement and loading to the loader class;
it only `instantiate`s and calls it. API systems simply omit device fields.

## Output cache

Pass `cache=<dir>` to enable an on-disk output cache. mushin stores system
outputs keyed by `(system_name, seed, sha256(input))` in JSONL files.
Subsequent runs replay cached outputs without calling the system — essential
for resuming after failures or re-scoring with a different metric:

```python
result = compare_llms(
    systems,
    data,
    metric=exact_match,
    seeds=range(5),
    cache="./eval_cache",
)
```

The cache stores **outputs only** (not metric scores) so you can freely change
the metric and re-run without re-calling the systems.

!!! warning "The cache cannot see inside your system"
    The key is the *dict name* plus seed and input — nothing about what the
    callable actually does. If you change the model, prompt template, or
    decoding parameters behind a name and reuse the same `cache=` dir, the old
    outputs are replayed and your comparison is silently stale. Any change to
    a system means a fresh cache directory (or a new system name).

## Pitfalls

- **Deterministic / seed-ignoring systems → false significance.** If a system
  ignores the seed (temperature 0, or an API call with no seed param), its
  scores are identical across all seeds — duplicated points, not independent
  samples. Two such systems with different means would otherwise get a tiny
  p-value (false significance), so `compare_llms` **warns** when it detects a
  system with identical scores across all seeds. Wire the seed to sampling
  (temperature, a provider seed) to get real variance, or treat that system's
  score as a single point estimate rather than a distribution.
- **Too few seeds for the chosen test.** The rank/paired tests can't reach
  p < 0.05 at small _n_ (a Wilcoxon over 3 seeds bottoms out at p = 0.25);
  `compare_llms` warns when the test you picked cannot reach `alpha` at the given
  seed count. Welch (the default) is fine at 3–5 seeds — still prefer ≥ 5 seeds
  for a more robust estimate.
- **Wrong output length.** A system must return exactly `len(inputs)` outputs
  in the same order; mushin raises `ValueError` immediately if it doesn't.
- **Seeds must be unique.** Each seed is one trial; a repeated seed is the same
  `(system, seed)` trial, not an independent sample, so duplicates would
  understate variance and inflate significance. `compare_llms` rejects a
  non-unique `seeds` with a `ValueError` (before any system runs).
- **Metric output names must not collide.** In a battery, a dict-returning
  metric expands to `<name>_<subkey>`. If that collides with another entry — e.g.
  `{"squad": SQuAD(), "squad_f1": custom}` produces two `squad_f1` — `compare_llms`
  raises a `ValueError` rather than silently overwriting one score. Rename the
  battery key to disambiguate.
- **Cache keys and input types.** The cache key is a **type-preserving** hash of
  the input, so distinct inputs never collide — `{1: "x"}` vs `{"1": "x"}`, or a
  tuple vs a list, hash differently (plain JSON would conflate them). Use
  JSON-friendly inputs (strings, numbers, lists, dicts); arbitrary objects fall
  back to `repr()`, which may not be stable across runs, so prefer simple strings
  or dicts as inputs for reliable cache hits.
- **Cached outputs must be JSON-serializable.** With `cache=`, system outputs
  are written as JSON; a non-serializable output (e.g. a custom object) raises a
  clear `TypeError`. Return strings or plain JSON-friendly values, or run
  without a cache.
- **Caching assumes per-input outputs.** On a partial cache hit, mushin calls
  the system on **only the missing inputs**, so a system's `output[i]` must
  depend solely on `input[i]` and the seed — not on which *other* inputs share
  the batch. This holds for the usual one-prompt-one-completion systems. If your
  system's per-item output depends on batch composition, don't use the cache.
  This also covers **duplicate inputs**: a compliant system returns the *same*
  output for the same `(input, seed)`, so the cache correctly serves one value
  for every occurrence. A system that returns *different* completions for
  repeated occurrences of one input under a single seed depends on
  occurrence/position — the same batch-composition dependence — so don't cache it
  (deduplicate the eval set, or run without a cache).

## See also

- [API Reference — llm](../reference/llm.md)
- [Comparing methods](compare.md) — the torch equivalent
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
