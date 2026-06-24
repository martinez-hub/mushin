# Docs & examples overhaul — Design

*Date: 2026-06-24*

## Goal

Make the documentation actually *teach* mushin. Today the guides describe the API
but demonstrate it only with undefined placeholders (`m0`, `m1`, `test_loader`,
`train_cnn`, `LitClassifier`) — nothing is copy-paste runnable. Replace that with
**complete, runnable, tested example scripts** that the guides embed verbatim,
deepen the guide prose, use **recognizable real data (MNIST)** in the scripts,
and add the missing conceptual pages. Builds on the docs site shipped in 0.3.0.

## Principle: examples are the single source of truth

Every code block a reader sees in a guide is **included from a tested script in
`examples/`** via `pymdownx.snippets` — so documentation code is literally the
code CI runs, and cannot drift from the API.

## Examples (`examples/`)

Each script is self-contained, uses real **MNIST**, and is split into a reusable
core + a `__main__` wrapper so it is both a runnable demo and unit-testable:

```python
# examples/compare_classifiers.py  (illustrative shape)
def make_model() -> torch.nn.Module: ...
def run(train_loader, test_loader, *, seeds=(0, 1, 2)) -> BenchmarkResult:
    # train `len(seeds)` copies of two small models, then mushin.benchmark.compare(...)
    ...
if __name__ == "__main__":
    train_loader, test_loader = load_mnist()   # torchvision MNIST
    print(run(train_loader, test_loader).summary())
```

Scripts:
- **`compare_classifiers.py`** — train two small CNNs across seeds on MNIST and
  `compare` them (classification battery + significance).
- **`study_mnist.py`** — `Study` (train + compare in one call) on MNIST.
- **`segmentation_demo.py`** — a *small synthetic* segmentation run (real seg data
  like VOC is too heavy for an example/test). The realistic torchvision
  dict-output `predict_fn` recipe is shown in the segmentation guide prose, not
  run.
- **`sweep_to_dataset.py`** — kept as-is (already tested).

### Tested without heavyweight CI

`tests/test_examples.py` imports each script's **core** (`run(...)`) and calls it
with a **tiny synthetic loader** (a handful of samples, 1–2 steps) — fast and
hermetic (no MNIST download in CI). The `__main__`/`load_mnist()` path is *not*
exercised in CI (it is standard torchvision); a module-level guard keeps
`load_mnist` import-safe so importing the script never triggers a download. The
mushin-specific logic (model construction, `compare`/`Study` wiring) is what the
test covers.

## Guides — deepen and de-placeholder

Every guide (`workflows`, `compare`, `study`, `segmentation`, `mcp`) is rewritten
to:
1. Open with a **concept intro** (the mental model / the "why").
2. Embed its runnable example via a snippet include, e.g.
   `--8<-- "examples/compare_classifiers.py:run"` (named sections delimited by
   `# --8<-- [start:run]` / `[end:run]` markers in the script). No more `m0`/`m1`.
3. Include an **annotated walkthrough of the output** — what `summary()`, the
   `(method × seed)` `xarray.Dataset`, and the significance columns mean.
4. End with a **Pitfalls & tips** admonition.

## New pages

Added to the nav under a new top-level section (Guides stays; these are broader):
- **Tutorial** (`docs/tutorial.md`) — one end-to-end narrative: define a sweep →
  collect a dataset → `compare` → read the statistics → interpret. Uses the MNIST
  examples.
- **Core concepts** (`docs/concepts.md`) — workflows, the `(method × seed)`
  dataset, statistical comparison, and the task registry seam.
- **Custom metrics & predict_fn** (`docs/guides/custom.md`) — extending mushin:
  custom `metrics` dict, custom `predict_fn` (generalizing the segmentation
  dict-output recipe), and adding a task.
- **Understanding the statistics** (`docs/guides/statistics.md`) — the tests
  (`welch`/`wilcoxon`/`mannwhitney`), Holm correction, effect size, the
  underpowered-test warning, and single-seed behavior. This is mushin's
  differentiator and deserves a dedicated page.

`mkdocs.yml` nav and `markdown_extensions` are updated: add `pymdownx.snippets`
with `base_path: ["."]` (repo root) so `examples/...` includes resolve, and
`check_paths: true` so a missing/renamed snippet fails the strict build.

## Data flow / build

`mkdocs build` → for each guide, `pymdownx.snippets` reads the referenced
`examples/*.py` section and inlines it as a fenced code block → mkdocstrings
renders the API reference as before → `--strict` fails on any missing snippet
path or broken link. So a renamed example function breaks the docs build (caught
in CI), keeping docs and code in lockstep.

## Testing strategy

- `tests/test_examples.py`: a smoke test per example core on synthetic data
  (fast, hermetic) — asserts it returns a `BenchmarkResult` with the expected
  dims/metrics, no crash.
- `mkdocs build --strict` (the existing `docs` PR job) now also validates every
  snippet include resolves (`check_paths: true`).
- Spot-check (manual, in the plan): each guide renders the embedded example code,
  not a placeholder; `--8<--` markers exist in the scripts for every include.

## Non-goals

- Heavy real datasets (VOC/ImageNet) in CI or examples.
- Jupyter notebooks; versioned docs; custom domain.
- Changing the public API (docs only — minor docstring fixes allowed if a symbol
  renders poorly).
- The LLM / agentic-eval workflows (a separate, later strategic effort).
