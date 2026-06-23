# Dogfood Regression Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five regression tests capturing behaviors that real dogfooding surfaced, so they can't silently regress. **No source changes.**

**Architecture:** Pure test additions under `tests/test_benchmark/`, matching existing style (small synthetic tensors, `TensorDataset`/`DataLoader`, deterministic seeds). Two tests in `test_compare.py` (public `compare()` API), three in `test_inference.py` (`evaluate`). Plus a `changes/` news fragment for the changelog gate. Spec: `docs/superpowers/specs/2026-06-23-dogfood-regression-tests-design.md`.

**Tech Stack:** pytest, torch, torchmetrics.

**Context for the implementer:** This is the `mushin` repo, in the worktree at `~/code/mushin/.worktrees/regression-tests`, on branch `dogfood-regression-tests` (off `main`). Run everything with `uv` (e.g. `uv run pytest ...`). **Never add Claude/AI authorship to any commit or file.** Every code block below is final — the two previously-uncertain values (the n=1 warning string and welch significance) were verified by running the real API; use them as written. For regression tests of *existing* behavior, the discipline is not "fail first" (they pass immediately) but the **mutation check**: after each test passes, temporarily break the thing it targets and confirm the test fails, then revert. This proves the test isn't vacuous.

Existing fixtures already in `tests/test_benchmark/test_compare.py` (reuse them): module-level `_loader(seed, n=64, d=4, num_classes=3)` and class `_Perfect(loader, num_classes=3)` (a model that reads targets off a fixed map → always correct).

---

### Task 1: `compare()` significance — single-seed warning + multi-seed positive

**Files:**
- Modify: `tests/test_benchmark/test_compare.py`

- [ ] **Step 1: Add `import pytest` at the top of the file**

The file currently starts with `import torch` then `from torch.utils.data import ...`. Add at the top, above `import torch`:

```python
import pytest
```

- [ ] **Step 2: Append the two tests to `tests/test_benchmark/test_compare.py`**

```python
def test_compare_single_seed_warns_underpowered_and_no_significance():
    # n=1: the underpowered warning (from the default wilcoxon) must propagate
    # through the public compare() API, and nothing may be flagged significant.
    # (The stats engine's n=1 NaN handling is covered separately in test_stats.)
    data = _loader(seed=0)
    with pytest.warns(UserWarning, match="cannot reach alpha"):
        result = compare(
            methods={"a": [_Perfect(data)], "b": [torch.nn.Linear(4, 3)]},
            data=data,
            task="classification",
            num_classes=3,
        )  # default test == wilcoxon
    assert result.data.sizes["seed"] == 1
    assert "accuracy" in result.data.data_vars
    assert not result.comparisons["significant"].any()


def test_compare_flags_significant_difference_end_to_end():
    # positive significance through the public compare(): clearly-better models
    # across several seeds -> the accuracy comparison is flagged significant with
    # the correct sign. (Only ever asserted on synthetic dicts at the
    # compare_methods level before.)
    data = _loader(seed=0)
    result = compare(
        methods={
            "good": [_Perfect(data) for _ in range(6)],
            "bad": [torch.nn.Linear(4, 3) for _ in range(6)],
        },
        data=data,
        task="classification",
        num_classes=3,
        test="welch",  # wilcoxon is underpowered at small n; welch can reach alpha
    )
    row = result.comparisons[result.comparisons["metric"] == "accuracy"].iloc[0]
    assert row["significant"]
    # mean_diff is method_a - method_b; identify the better method sign-robustly.
    better = row["method_a"] if row["mean_diff"] > 0 else row["method_b"]
    assert better == "good"
```

- [ ] **Step 3: Run the two new tests — expect PASS**

Run: `uv run pytest tests/test_benchmark/test_compare.py::test_compare_single_seed_warns_underpowered_and_no_significance tests/test_benchmark/test_compare.py::test_compare_flags_significant_difference_end_to_end -q`
Expected: 2 passed. (scipy may print RuntimeWarnings at n=1 — harmless; `pytest.warns` only requires the matching `UserWarning`.)

- [ ] **Step 4: Mutation check (prove non-vacuous), then revert**

1. In `test_compare_single_seed_...`, change `num_classes=3,` call to pass `test="welch"` (welch is silent at n=1). Re-run: it must **FAIL** (`DID NOT WARN`). Revert.
2. In `test_compare_flags_significant_difference_end_to_end`, change `bad` to `[_Perfect(data) for _ in range(6)]` (tie). Re-run: it must **FAIL** (`assert row["significant"]` is False for tied methods). Revert.

Run after reverting: `uv run pytest tests/test_benchmark/test_compare.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark/test_compare.py
git commit -m "test: cover single-seed warning propagation and multi-seed significance through compare()"
```

---

### Task 2: dict-output model routed through a custom `predict_fn`

**Files:**
- Modify: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Append the test to `tests/test_benchmark/test_inference.py`**

```python
def test_evaluate_with_dict_output_model_and_custom_predict_fn():
    # Real models (e.g. torchvision segmentation) return a dict {"out": logits},
    # not a tensor, so the default predict_fn can't be used. A custom predict_fn
    # extracting ["out"] must flow through evaluate end-to-end. This locks in the
    # segmentation-dogfood friction.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    mapping = {tuple(xi.tolist()): int(yi) for xi, yi in zip(x, y)}

    class DictModel(torch.nn.Module):
        def forward(self, xb):
            idx = torch.tensor([mapping[tuple(r.tolist())] for r in xb])
            logits = torch.nn.functional.one_hot(idx, 3).float() * 10.0
            return {"out": logits}  # dict output, like torchvision seg models

    def predict(model, xb):
        logits = model(xb)["out"]
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs

    out = evaluate(
        DictModel(),
        loader,
        classification_battery(3),
        predict,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    assert out["accuracy"] == 1.0
```

- [ ] **Step 2: Run it — expect PASS**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_evaluate_with_dict_output_model_and_custom_predict_fn -q`
Expected: 1 passed.

- [ ] **Step 3: Mutation check, then revert**

Temporarily replace `predict` in the `evaluate(...)` call with `default_classification_predict_fn` (add `from mushin.benchmark._predict import default_classification_predict_fn`). Re-run: it must **FAIL** (the default calls `softmax` on a dict → `TypeError`), proving the custom predict_fn seam is what makes dict-output models work. Revert.

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark/test_inference.py
git commit -m "test: a dict-output model flows through evaluate via a custom predict_fn"
```

---

### Task 3: streaming segmentation `ignore_index` excludes void across batches

**Files:**
- Modify: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Append the test**

```python
def test_evaluate_segmentation_ignores_void_across_batches():
    # ignore_index must exclude void pixels through the *streaming* evaluate path
    # (the existing test only covers the one-shot compute_battery). Model is
    # correct on every non-void pixel; its arbitrary void prediction must not
    # count because the void label is excluded.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import segmentation_battery
    from mushin.benchmark._predict import default_segmentation_predict_fn

    g = torch.Generator().manual_seed(0)
    N, C, H, W = 8, 3, 6, 6
    x = torch.randn(N, 1, H, W, generator=g)
    true = torch.randint(0, C, (N, H, W), generator=g)  # the real classes
    target = true.clone()
    target[:, 0, 0] = 255  # one void pixel per image
    loader = DataLoader(TensorDataset(x, target), batch_size=4)  # 2 batches
    mapping = {tuple(xi.flatten().tolist()): t for xi, t in zip(x, true)}

    class PerfectSeg(torch.nn.Module):
        def forward(self, xb):
            outs = []
            for xi in xb:
                t = mapping[tuple(xi.flatten().tolist())]  # predict the true class
                outs.append(
                    torch.nn.functional.one_hot(t, C).permute(2, 0, 1).float() * 10
                )
            return torch.stack(outs)

    res = evaluate(
        PerfectSeg(),
        loader,
        segmentation_battery(C, ignore_index=255),
        default_segmentation_predict_fn,
        prob_metrics=frozenset(),
    )
    assert res["pixel_acc"] == 1.0
    assert res["miou"] == 1.0
```

- [ ] **Step 2: Run it — expect PASS**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_evaluate_segmentation_ignores_void_across_batches -q`
Expected: 1 passed.

- [ ] **Step 3: Mutation check, then revert**

Temporarily drop `ignore_index=255` (use `segmentation_battery(C)`). Re-run: it must **FAIL** (the void label 255 is now scored against `num_classes=3` — `pixel_acc` drops below 1.0 or torchmetrics raises). Revert.

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark/test_inference.py
git commit -m "test: streaming evaluate excludes void pixels via segmentation ignore_index"
```

---

### Task 4: `evaluate` accepts explicit device and resets metric state between calls

**Files:**
- Modify: `tests/test_benchmark/test_inference.py`

- [ ] **Step 1: Append the test**

```python
def test_evaluate_explicit_device_and_resets_between_calls():
    # evaluate(device=...) must run on the given device, and reusing the same
    # battery dict across calls must NOT leak metric state (evaluate resets each
    # metric before the batch loop).
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    def loader(seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(16, 4, generator=g)
        y = torch.randint(0, 3, (16,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=8)

    model = torch.nn.Linear(4, 3)
    cpu = torch.device("cpu")
    pm = frozenset({"auroc", "ece"})

    battery = classification_battery(3)
    evaluate(model, loader(1), battery, default_classification_predict_fn,
             prob_metrics=pm, device=cpu)  # first call dirties the metric state
    reused = evaluate(model, loader(2), battery, default_classification_predict_fn,
                      prob_metrics=pm, device=cpu)  # same battery, different data
    fresh = evaluate(model, loader(2), classification_battery(3),
                     default_classification_predict_fn, prob_metrics=pm, device=cpu)

    assert reused.keys() == fresh.keys()
    for k in reused:
        assert abs(reused[k] - fresh[k]) < 1e-6  # no carryover from the first call
```

- [ ] **Step 2: Run it — expect PASS**

Run: `uv run pytest tests/test_benchmark/test_inference.py::test_evaluate_explicit_device_and_resets_between_calls -q`
Expected: 1 passed.

- [ ] **Step 3: Mutation check, then revert**

In `src/mushin/benchmark/_inference.py`, temporarily comment out the `metric.reset()` line inside the `for metric in battery.values():` loop. Re-run the test: it must **FAIL** (`reused` now carries call-1 state and differs from `fresh`). **Restore the line** — this is a source file; it must be reverted exactly (the task makes no source changes).

Run after reverting: `uv run pytest tests/test_benchmark/test_inference.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark/test_inference.py
git commit -m "test: evaluate honors explicit device and resets metric state between calls"
```

---

### Task 5: changelog fragment + whole-suite verification

**Files:**
- Create: `changes/+dogfood-regression-tests.misc.md`

- [ ] **Step 1: Add the news fragment**

These are internal test-only additions (nothing user-facing), so use a `misc`
fragment to satisfy the changelog gate without adding a changelog line. Create
`changes/+dogfood-regression-tests.misc.md` as an **empty** file:

```bash
: > changes/+dogfood-regression-tests.misc.md
```

- [ ] **Step 2: Confirm the changelog gate passes**

Run: `uv run towncrier check --compare-with main`
Expected: passes (a fragment is present on the branch).

- [ ] **Step 3: Run the full local gate**

Run: `make check`
Expected: ruff lint + format-check + codespell + the full pytest suite all pass. The five new tests are included; nothing else changed.

- [ ] **Step 4: Commit**

```bash
git add changes/+dogfood-regression-tests.misc.md
git commit -m "docs: add news fragment for the dogfood regression tests"
```

---

## Notes for execution

- After all tasks: push the branch and open a PR against `main`. This PR adds **no** `CHANGELOG.md` edit, so it needs **no** `changelog-exempt` label — the `misc` fragment satisfies the gate normally.
- Then **check the Codex connector's review** and address any flags before merge; validate green CI across the matrix.
- No torch/numpy code paths change, so the Linux-container (Docker) torch check is not required — note that in the PR.
