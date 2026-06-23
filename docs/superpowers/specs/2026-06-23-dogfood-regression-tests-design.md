# Dogfood regression tests — Design

*Date: 2026-06-23*

## Goal

Lock in, as tests, four behaviors that real dogfooding of `compare` / `Study` /
segmentation surfaced — so they can't silently regress. This is **Piece 1** of
the testing expansion (Piece 2, the CI matrix: Windows + min-version, is a
separate spec). Pure test additions: **no source changes**.

## Scope and existing coverage

Each test is checked against what already exists so we **complement, not
duplicate**:

| Behavior | Existing coverage | This spec adds |
| --- | --- | --- |
| Single-seed `compare` | none (`test_compare_end_to_end` uses 3 seeds) | a 1-seed run: metrics present, underpowered warning fires, no significance, no crash |
| Dict-output model + custom `predict_fn` | none (all tests use tensor-returning models + default predict_fn) | a model returning `{"out": logits}` routed through a custom predict_fn |
| Segmentation `ignore_index` | `test_segmentation_battery_ignore_index` tests the **one-shot** `compute_battery` only | the **streaming** `evaluate` path excludes void across batches |
| `evaluate` device / state | `test_evaluate_streams_and_matches_one_shot` (numerics only) | explicit `device=` run + reset-across-calls (no state leak) |

## Owned vs. delegated

Nothing new owned; these tests exercise existing owned code (`compare`,
`evaluate`, `segmentation_battery`) and the documented `predict_fn` seam.

## Test designs

All live under `tests/test_benchmark/`, matching the existing style (small
synthetic tensors, `TensorDataset`/`DataLoader`, deterministic
`torch.Generator().manual_seed`). Reference signatures:
`evaluate(model, data, battery, predict_fn, prob_metrics, device=None)`;
`compare(methods, data, task="classification", *, num_classes=None,
predict_fn=None, ..., test="wilcoxon", alpha=0.05, prob_metrics=None,
ignore_index=None, device=None)`; `segmentation_battery(num_classes,
ignore_index=None)`.

### 1. Single-seed `compare` — `tests/test_benchmark/test_compare.py`

`test_compare_single_seed_warns_and_reports_no_significance`:
- Build a re-iterable classification loader (reuse the file's `_loader`).
- `compare(methods={"a": [m_a], "b": [m_b]}, data=..., task="classification",
  num_classes=3, test="welch")` — **one model per method** (n_seeds = 1).
- Wrap the call in `pytest.warns(UserWarning, match=...)` asserting the
  underpowered-test warning fires (the n=1 case can never reach `alpha`).
- Assert: `result.data.sizes["seed"] == 1`; metric vars present;
  `not result.comparisons["significant"].any()` (no false positive at n=1);
  no exception.
- **Plan must verify** which test/category emits the warning at n=1 (parametric
  tests yield NaN p at n=1; `warn_if_underpowered` is driven by test + n_seeds).
  Pin the exact `match=` string and the test (`welch` vs `wilcoxon`) after
  running it. If `welch` does not warn at n=1, use the test that does and assert
  the no-significance behavior for the chosen test.

### 2. Dict-output model + custom `predict_fn` — `tests/test_benchmark/test_inference.py`

`test_evaluate_with_dict_output_model_and_custom_predict_fn`:
- A tiny model whose `forward` returns a **dict**: `{"out": logits}` of shape
  `(N, 3)` (mirrors torchvision segmentation models, which the real dogfood hit).
- A custom predict_fn:
  ```python
  def predict(model, x):
      logits = model(x)["out"]
      probs = torch.softmax(logits, dim=-1)
      return probs.argmax(dim=-1), probs
  ```
- Run through `evaluate(model, loader, classification_battery(3), predict,
  prob_metrics=frozenset({"auroc", "ece"}))`.
- Use a model that returns logits encoding the true label (one-hot * large) so
  the result is deterministic → assert `accuracy == 1.0`.
- The point: this path is impossible with the default predict_fn (which would
  call `softmax` on a dict and raise) — proving the `predict_fn` seam is what
  makes real models work. A short comment records that.

### 3. Streaming segmentation `ignore_index` — `tests/test_benchmark/test_inference.py`

`test_evaluate_segmentation_ignores_void_across_batches`:
- A 2-batch loader of `(x, mask)` with masks shape `(N, H, W)` containing a void
  label `255` in some pixels.
- A model returning per-pixel logits `(N, C, H, W)` that is **correct on every
  non-void pixel** and arbitrary on void pixels.
- `evaluate(model, loader, segmentation_battery(3, ignore_index=255),
  default_segmentation_predict_fn, prob_metrics=frozenset())`.
- Assert `pixel_acc == 1.0` and `miou == 1.0` — void excluded, so the arbitrary
  void predictions don't hurt the score. (Complements the existing one-shot
  `compute_battery` test by exercising the streaming accumulation across
  batches.)

### 4. `evaluate` device + state reset — `tests/test_benchmark/test_inference.py`

`test_evaluate_explicit_device_and_resets_between_calls`:
- Build one `classification_battery(3)` and call `evaluate(..., device=torch.device("cpu"))`
  on loader A, then on loader B (different data) **with the same battery dict**.
- Assert the second result equals `evaluate` of a **fresh** battery on loader B
  (proves `metric.reset()` prevents carryover from call A).
- Asserts the explicit `device=` argument is accepted and the run completes on
  CPU. (CI is CPU-only; this is a correctness/no-leak test, not a GPU test.)

## Error handling

These are tests; "error handling" is that each test would **fail loudly** if the
behavior regressed: e.g. test 2 fails (raises) if the default predict_fn path
were used; test 4 fails if metric state leaked across calls.

## Testing / verification strategy

- Each new test must pass: `uv run pytest tests/test_benchmark -q`.
- **Mutation sanity (manual, during the plan):** confirm each test actually
  exercises its target — e.g. temporarily swap the custom predict_fn for the
  default in test 2 and confirm it fails; drop `ignore_index` in test 3 and
  confirm `pixel_acc < 1.0`. This guards against vacuous (always-green) tests.
- Full suite stays green: `make check`.
- A `changes/<id>.added.md` fragment is required (the new changelog gate).

## Non-goals

- The CI matrix expansion (Windows runner, min-version job, any resulting floor
  bumps) — that is Piece 2, a separate spec/plan/PR.
- No new source behavior, no new public API.
- No GPU/device-transfer testing beyond CPU correctness.
