# Docs & Examples Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace placeholder doc snippets with complete, runnable, tested example scripts that the guides embed; deepen guide prose; add conceptual pages. Doc code becomes the tested code (via `pymdownx.snippets`).

**Architecture:** `examples/*.py` scripts each expose a reusable `run(...)` core (tested on synthetic data) + a `__main__` that wires real MNIST. Guides include named sections from those scripts with `--8<-- "examples/<file>.py:<section>"`. `mkdocs build --strict` (with `check_paths: true`) fails if a snippet path/section is missing. Spec: `docs/superpowers/specs/2026-06-24-docs-examples-overhaul-design.md`.

**Tech Stack:** MkDocs Material, pymdownx.snippets, pytest, torch.

**Context for the implementer:** Repo `mushin` in the worktree at `/Users/josuemartinez/code/mushin/.worktrees/docs-overhaul`, branch `docs-overhaul` (off `main`). Run with `uv`. **Never add Claude/AI authorship to any commit or file.** Snippet section-extraction is verified working: markers are `# --8<-- [start:NAME]` / `# --8<-- [end:NAME]` in the `.py`, included as `--8<-- "examples/<file>.py:NAME"` inside a fenced ```python block. After each task run `uv run --group docs mkdocs build --strict` (passes; the Material "MkDocs 2.0" banner is informational, not an error) and the relevant `pytest`, then commit.

Verify the public API by reading source before writing examples: `src/mushin/benchmark/compare.py` (signature: `compare(methods, data, task="classification", *, num_classes=None, predict_fn=None, metrics=None, prob_metrics=None, test="wilcoxon", alpha=0.05, ignore_index=None, device=None)`), `src/mushin/study/_study.py` (`Study(methods, load_fn, seeds, data, *, num_classes, task="classification", test="wilcoxon", alpha=0.05, ignore_index=None, working_dir=None)`), and `examples/sweep_to_dataset.py` (existing tested example) + `tests/test_examples.py`.

---

### Task 1: mkdocs.yml — snippets extension + nav for new pages

**Files:**
- Modify: `mkdocs.yml`
- Create: `docs/tutorial.md`, `docs/concepts.md`, `docs/guides/custom.md`, `docs/guides/statistics.md` (stubs)

- [ ] **Step 1: Add `pymdownx.snippets` to `markdown_extensions`** in `mkdocs.yml`:

```yaml
  - pymdownx.snippets:
      base_path: ["."]
      check_paths: true
```

- [ ] **Step 2: Update the `nav`** to add the new pages (place Tutorial + Core concepts before Guides; add the two new guide pages):

```yaml
nav:
  - Home: index.md
  - Install: install.md
  - Quickstart: quickstart.md
  - Tutorial: tutorial.md
  - Core concepts: concepts.md
  - Guides:
      - Workflows & sweeps: guides/workflows.md
      - Comparing methods: guides/compare.md
      - Studies: guides/study.md
      - Segmentation: guides/segmentation.md
      - Custom metrics & predict_fn: guides/custom.md
      - Understanding the statistics: guides/statistics.md
      - Analyzing from Claude Code: guides/mcp.md
  - API Reference:
      - benchmark: reference/benchmark.md
      - study: reference/study.md
      - workflows: reference/workflows.md
      - lightning: reference/lightning.md
      - utils: reference/utils.md
  - Contributing: contributing.md
```

- [ ] **Step 3: Create stub pages** so `--strict` passes now: `docs/tutorial.md`,
  `docs/concepts.md`, `docs/guides/custom.md`, `docs/guides/statistics.md`, each
  `# Title` + one sentence (filled in Tasks 5–6).

- [ ] **Step 4: Verify + commit**

Run: `uv run --group docs mkdocs build --strict` → passes.
```bash
git add mkdocs.yml docs/tutorial.md docs/concepts.md docs/guides/custom.md docs/guides/statistics.md
git commit -m "docs: add pymdownx.snippets and nav entries for new pages"
```

---

### Task 2: `examples/compare_classifiers.py` + test

**Files:**
- Create: `examples/compare_classifiers.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Write the example** `examples/compare_classifiers.py`:

```python
"""Compare two small classifiers on MNIST with statistical significance.

Run it (downloads MNIST):  python examples/compare_classifiers.py

The reusable `run()` core is exercised by the test suite on tiny synthetic data,
so CI never downloads MNIST.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mushin.benchmark import BenchmarkResult, compare


# --8<-- [start:models]
def small_cnn() -> nn.Module:
    """A tiny convolutional classifier for 1x28x28 images."""
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10),
    )


def mlp() -> nn.Module:
    """A tiny fully-connected classifier."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
# --8<-- [end:models]


def _train(model: nn.Module, loader: DataLoader, epochs: int = 1) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    return model.eval()


# --8<-- [start:run]
def run(
    train_loader: DataLoader, test_loader: DataLoader, *, seeds=(0, 1, 2)
) -> BenchmarkResult:
    """Train one CNN and one MLP per seed, then compare them with statistics."""
    methods: dict[str, list[nn.Module]] = {"cnn": [], "mlp": []}
    for seed in seeds:
        torch.manual_seed(seed)
        methods["cnn"].append(_train(small_cnn(), train_loader))
        methods["mlp"].append(_train(mlp(), train_loader))

    return compare(
        methods,
        data=test_loader,
        task="classification",
        num_classes=10,
        test="welch",
    )
# --8<-- [end:run]


def _load_mnist(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


if __name__ == "__main__":
    train_loader, test_loader = _load_mnist()
    result = run(train_loader, test_loader)
    print(result.summary().to_string(index=False))
```

- [ ] **Step 2: Add a synthetic smoke test** to `tests/test_examples.py`:

```python
def test_compare_classifiers_example_runs_on_synthetic():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from compare_classifiers import run
    from mushin.benchmark import BenchmarkResult

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (32,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    result = run(loader, loader, seeds=(0, 1))
    assert isinstance(result, BenchmarkResult)
    assert result.data.sizes["seed"] == 2
    assert "accuracy" in result.data.data_vars
```
(`tests/test_examples.py` works because `pyproject.toml` sets `pythonpath = ["examples"]`, so `from compare_classifiers import run` resolves. Confirm that line still exists in `[tool.pytest.ini_options]`.)

- [ ] **Step 3: Run the test + a strict build**

Run: `uv run pytest tests/test_examples.py::test_compare_classifiers_example_runs_on_synthetic -q`
Expected: 1 passed (it must NOT download MNIST — importing the module must not call `_load_mnist`; the guard is the `if __name__ == "__main__"` block).
Run: `uv run --group docs mkdocs build --strict` → still passes.

- [ ] **Step 4: Commit**

```bash
git add examples/compare_classifiers.py tests/test_examples.py
git commit -m "examples: runnable MNIST compare example with synthetic smoke test"
```

---

### Task 3: `examples/study_mnist.py` + test

**Files:**
- Create: `examples/study_mnist.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Write the example.** Mirror Task 2's structure but for `Study`.
  Provide a `run(data_loader, *, seeds=(0, 1, 2), working_dir)` core that builds a
  `Study` with `train_fn(seed) -> checkpoint_path` and `load_fn(path) -> model`,
  marked with `# --8<-- [start:run]` / `[end:run]`, and a `# --8<-- [start:train_fn]`
  section showing the `train_fn`/`load_fn` definitions. Read
  `src/mushin/study/_study.py` and `tests/test_study/` for the exact `Study`
  contract (`methods` maps name → `train_fn`; `load_fn`; `seeds`; `data`;
  keyword-only `num_classes`; optional `working_dir`). The `__main__` wires MNIST
  via a `_load_mnist()` like Task 2. Keep models tiny (reuse the `small_cnn`/`mlp`
  shapes). The `train_fn` should train briefly and save a checkpoint under
  `working_dir`, returning its path.

- [ ] **Step 2: Add a synthetic smoke test** to `tests/test_examples.py` that calls
  the `study_mnist.run(...)` core with a tiny synthetic loader and a `tmp_path`
  working dir, and asserts it returns a `BenchmarkResult` with `seed` dim ==
  number of seeds. Use `tmp_path` (pytest fixture) for `working_dir`.

- [ ] **Step 3: Run the test + strict build**

Run: `uv run pytest tests/test_examples.py -q` → all pass (no MNIST download).
Run: `uv run --group docs mkdocs build --strict` → passes.

- [ ] **Step 4: Commit**

```bash
git add examples/study_mnist.py tests/test_examples.py
git commit -m "examples: runnable MNIST Study example with synthetic smoke test"
```

---

### Task 4: `examples/segmentation_demo.py` + test

**Files:**
- Create: `examples/segmentation_demo.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Write the example** — a *synthetic* segmentation demo (no heavy
  data). Model: a tiny `nn.Conv2d(in_ch, num_classes, 1)` that maps `(N, C_in, H, W)`
  → per-pixel logits `(N, num_classes, H, W)`. Provide a `run(loader, *, num_classes,
  seeds=(0,1,2))` core marked `# --8<-- [start:run]` that calls
  `compare(..., task="segmentation", num_classes=num_classes)`. Also include a
  `# --8<-- [start:dict_predict]` section with the torchvision dict-output
  `predict_fn` recipe (the function only — it's referenced by the segmentation
  guide, not executed):
  ```python
  # --8<-- [start:dict_predict]
  def torchvision_seg_predict(model, x):
      """Adapt a torchvision segmentation model (returns {"out": logits})."""
      logits = model(x)["out"]
      probs = logits.softmax(dim=1)
      return probs.argmax(dim=1), probs
  # --8<-- [end:dict_predict]
  ```
  The `__main__` builds a small synthetic `(N, C, H, W)` dataset + `(N, H, W)`
  masks and prints `run(...).summary()`.

- [ ] **Step 2: Add a synthetic smoke test** asserting `run(...)` returns a
  `BenchmarkResult` with `"miou"` in `result.data.data_vars`.

- [ ] **Step 3: Run + strict build**

Run: `uv run pytest tests/test_examples.py -q` → all pass.
Run: `uv run --group docs mkdocs build --strict` → passes.

- [ ] **Step 4: Commit**

```bash
git add examples/segmentation_demo.py tests/test_examples.py
git commit -m "examples: synthetic segmentation demo with dict-output predict_fn recipe"
```

---

### Task 5: Rewrite the existing guides to embed examples + deepen prose

**Files:**
- Modify: `docs/guides/compare.md`, `docs/guides/study.md`, `docs/guides/segmentation.md`, `docs/guides/workflows.md`

For each guide, follow this structure: (1) concept intro paragraph; (2) the
runnable example embedded via a snippet include (replacing ALL `m0/m1/test_loader`
placeholders); (3) an annotated walkthrough of the output (`summary()`, the
`(method × seed)` `xarray.Dataset`, the significance columns); (4) a
`!!! tip "Pitfalls"` admonition.

- [ ] **Step 1: `compare.md`** — embed the compare example:
  ````markdown
  ```python
  --8<-- "examples/compare_classifiers.py:run"
  ```
  ````
  Keep the existing statistics/battery prose but correct it (the classification
  battery is accuracy, macro F1, macro precision/recall, AUROC, ECE; segmentation
  adds Dice — verify against `src/mushin/benchmark/_metrics.py`).

- [ ] **Step 2: `study.md`** — embed `--8<-- "examples/study_mnist.py:train_fn"`
  and `--8<-- "examples/study_mnist.py:run"`; keep the parameter table (verify it
  against `_study.py`).

- [ ] **Step 3: `segmentation.md`** — embed `--8<-- "examples/segmentation_demo.py:run"`
  for the basic flow and `--8<-- "examples/segmentation_demo.py:dict_predict"` for
  the torchvision recipe (this replaces the hand-written recipe). Keep the
  `ignore_index` and 2-tuple-`predict_fn` notes.

- [ ] **Step 4: `workflows.md`** — embed the relevant section from the existing
  `examples/sweep_to_dataset.py` (add `# --8<--` markers to that script around the
  workflow definition + run, in the same commit) so the guide shows the tested
  sweep code instead of a fragment.

- [ ] **Step 5: Verify + commit**

Run: `uv run --group docs mkdocs build --strict` → passes (every include resolves;
`check_paths: true` would fail a typo'd section name). Spot-check that
`site/guides/compare/index.html` contains real code (`def run(` and `def small_cnn(`),
not `m0`.
```bash
git add docs/guides/compare.md docs/guides/study.md docs/guides/segmentation.md docs/guides/workflows.md examples/sweep_to_dataset.py
git commit -m "docs: guides embed runnable examples and gain annotated output + pitfalls"
```

---

### Task 6: New pages — Tutorial, Core concepts, Custom, Statistics

**Files:**
- Modify: `docs/tutorial.md`, `docs/concepts.md`, `docs/guides/custom.md`, `docs/guides/statistics.md`

- [ ] **Step 1: `tutorial.md`** — one end-to-end narrative: define a sweep
  (embed from `sweep_to_dataset.py`), collect the `(lr × seed)` dataset, then run
  `compare` (embed from `compare_classifiers.py:run`), then read the statistics.
  Link onward to the guides and Core concepts.

- [ ] **Step 2: `concepts.md`** — the mental model: workflows (sweep → dataset),
  the `(method × seed)` `xarray.Dataset`, statistical comparison (why seeds +
  significance), and the task registry seam (`task="classification"|"segmentation"`).
  No new code — prose + small inline snippets are fine; link to reference pages.

- [ ] **Step 3: `guides/custom.md`** — extending mushin: a custom `metrics` dict,
  a custom `predict_fn` (embed `--8<-- "examples/segmentation_demo.py:dict_predict"`
  as the worked example and generalize it), and a note on `prob_metrics`. Verify
  the `metrics`/`predict_fn`/`prob_metrics` kwargs against `compare.py`.

- [ ] **Step 4: `guides/statistics.md`** — the tests (`welch`, `wilcoxon`,
  `mannwhitney`), Holm correction, effect size, the underpowered-test warning, and
  single-seed behavior (NaN p-value → not significant). Verify names/behavior
  against `src/mushin/benchmark/_stats.py`. This is the differentiator page.

- [ ] **Step 5: Verify + commit**

Run: `uv run --group docs mkdocs build --strict` → passes.
```bash
git add docs/tutorial.md docs/concepts.md docs/guides/custom.md docs/guides/statistics.md
git commit -m "docs: add tutorial, core concepts, custom metrics/predict_fn, and statistics pages"
```

---

### Task 7: News fragment + whole-branch verification

**Files:**
- Create: `changes/+docs-examples-overhaul.added.md`

- [ ] **Step 1: Fragment**

`changes/+docs-examples-overhaul.added.md`:
```
Overhauled the documentation: runnable, tested example scripts (MNIST) that the guides embed verbatim, deeper guides with annotated output, and new Tutorial, Core concepts, Custom metrics/predict_fn, and Statistics pages.
```

- [ ] **Step 2: Whole-branch verification**

Run: `uv run pytest tests/test_examples.py -q` → all example smoke tests pass.
Run: `make check` → lint + format + spell + full suite pass (examples are on
`pythonpath`, so they are import-checked).
Run: `uv run --group docs mkdocs build --strict` → passes.
Run: `uv run towncrier check --compare-with main` → fragment found.
Confirm no `m0`/`b0`/`test_loader` placeholders remain in guides:
`grep -rnE "\b(m0|m1|b0|b1|test_loader|train_cnn)\b" docs/ && echo "PLACEHOLDERS REMAIN (bad)" || echo "clean"`

- [ ] **Step 3: Commit**

```bash
git add changes/+docs-examples-overhaul.added.md
git commit -m "docs: news fragment for the docs and examples overhaul"
```

---

## Notes for execution

- After all tasks: push (`git -c http.postBuffer=524288000 push -u origin docs-overhaul` if a plain push stalls) and open a PR. The `docs` strict-build job + the test matrix + changelog gate must be green. Check Codex and address flags.
- No application code changes (docs + examples + tests only); the Docker/torch check is not required.
- `examples/` are import-checked by the test suite (pythonpath) and embedded by the docs, so a broken example fails BOTH the tests and the strict build.
