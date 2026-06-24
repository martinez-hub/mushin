# LLM evaluation

The "vibes eval" problem: it's common to eyeball a handful of outputs and
declare a new prompt or model "better" — but that impression may be pure
sampling noise. `mushin.llm.compare_llms` runs each system across
reproducible stochastic seeds, scores every `(system, seed)` pair with a
user-supplied metric, and reports Holm-corrected pairwise significance. The
same statistical spine that powers `mushin.benchmark.compare` for torch
models works here — you just bring the systems, data, and metric.

## Quickstart

```python
--8<-- "examples/compare_llms_demo.py:run"
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

```python
result = compare_llms(
    {"gpt4": gpt4_system, "claude": claude_system},
    data=eval_data,
    metric=exact_match,
    seeds=range(10),   # 10 seeds → more power
    test="welch",
)
```

Use more seeds (≥ 5) for meaningful significance. With Welch's t-test and 5
seeds you get reasonable power; with 3 seeds even a clear winner may not reach
p < 0.05 — the tool will warn you.

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
    data,                 # each example's `reference` is a plain string
    metric={"wer": WordErrorRate(), "cer": CharErrorRate()},
    seeds=range(5),
)
```

A metric whose `compute()` returns a **dict** (e.g. `SQuAD` → `exact_match`,
`f1`) expands into one data variable per key, named `<metric_name>_<subkey>`.

!!! warning "Shape `output`/`reference` to the metric"
    mushin passes your raw `output`s and `reference`s straight to
    `metric.update(outputs, references)` — it does **not** reshape them. Each
    torchmetrics text metric expects a specific shape, so shape your example
    `reference` (and `output`) accordingly:

    | metric | `output` | `reference` (per example) |
    |---|---|---|
    | `WordErrorRate`, `CharErrorRate`, `MatchErrorRate`, `Perplexity` | `str` | `str` |
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
    MyHFGenerator,           # your class wrapping HF from_pretrained
    model_name="mistralai/Mistral-7B-v0.1",
    device_map="auto",       # shard across CPU/GPU automatically
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
    systems, data, metric=exact_match, seeds=range(5),
    cache="./eval_cache",
)
```

The cache stores **outputs only** (not metric scores) so you can freely change
the metric and re-run without re-calling the systems.

## Pitfalls

- **Deterministic / seed-ignoring systems → false significance.** If a system
  ignores the seed (temperature 0, or an API call with no seed param), its
  scores are identical across all seeds — duplicated points, not independent
  samples. Two such systems with different means would otherwise get a tiny
  p-value (false significance), so `compare_llms` **warns** when it detects a
  system with identical scores across all seeds. Wire the seed to sampling
  (temperature, a provider seed) to get real variance, or treat that system's
  score as a single point estimate rather than a distribution.
- **Too few seeds.** With `seeds=range(3)` even a clear winner may not reach
  p < 0.05. mushin warns you when this happens. Use ≥ 5 seeds or switch to
  `test="welch"`.
- **Wrong output length.** A system must return exactly `len(inputs)` outputs
  in the same order; mushin raises `ValueError` immediately if it doesn't.
- **Cache key collisions.** The cache key is `sha256(json(input))`. If your
  inputs are objects that don't serialize cleanly to JSON, use simple strings
  or dicts as inputs.
- **Cached outputs must be JSON-serializable.** With `cache=`, system outputs
  are written as JSON; a non-serializable output (e.g. a custom object) raises a
  clear `TypeError`. Return strings or plain JSON-friendly values, or run
  without a cache.
- **Caching assumes per-input outputs.** On a partial cache hit, mushin calls
  the system on **only the missing inputs**, so a system's `output[i]` must
  depend solely on `input[i]` and the seed — not on which *other* inputs share
  the batch. This holds for the usual one-prompt-one-completion systems. If your
  system's per-item output depends on batch composition, don't use the cache.

## See also

- [API Reference — llm](../reference/llm.md)
- [Comparing methods](compare.md) — the torch equivalent
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
