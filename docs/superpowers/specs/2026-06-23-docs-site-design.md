# Documentation website for mushin-py — Design

*Date: 2026-06-23*

## Goal

Ship a real documentation website (like rai-toolbox / hydra-zen) that teaches
people how to use mushin — both narrative how-to guides and an auto-generated
API reference for every public function/class. Close the gap between "a package
on PyPI" and "a library people can actually learn and adopt."

## Toolchain

[MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
+ [mkdocstrings[python]](https://mkdocstrings.github.io/) (griffe backend,
NumPy-style docstrings). Chosen for low authoring friction (markdown-native — the
repo is already markdown-heavy), a polished default theme, and automatic API docs
from the existing NumPy docstrings.

- `mkdocs.yml` at the repo root.
- Content under `docs/` (markdown). The existing `docs/superpowers/` (internal
  specs/plans) is **excluded** from the site via `mkdocs.yml` `exclude_docs` /
  nav scoping so it never publishes.
- A new PEP 735 `docs` dependency group in `pyproject.toml`:
  `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-gen-files` (optional, only if
  needed for reference generation; default is hand-written reference stubs).

## Site structure (nav)

```
Home                      docs/index.md
Install                   docs/install.md
Quickstart                docs/quickstart.md
Guides/
  Workflows & sweeps      docs/guides/workflows.md
  Comparing methods       docs/guides/compare.md
  Studies                 docs/guides/study.md
  Segmentation            docs/guides/segmentation.md
  Analyzing from Claude    docs/guides/mcp.md
API Reference/
  benchmark               docs/reference/benchmark.md
  study                   docs/reference/study.md
  workflows               docs/reference/workflows.md
  lightning               docs/reference/lightning.md
  utils                   docs/reference/utils.md
Contributing              docs/contributing.md
```

### Page responsibilities

- **Home** — the value proposition (mushin owns the *evaluate + report* layer on
  top of Lightning + hydra-zen), the logo, and quick links to Install/Quickstart/
  Guides. Reuses the README's "What it provides" framing.
- **Install** — `pip install mushin-py` → `import mushin`; extras (`viz`,
  `netcdf`); and an **honest** support matrix (Python 3.9–3.14; torch ≥ 2.4 /
  Lightning ≥ 2.4 non-Intel, torch 2.2.x on Intel-macOS) taken from the
  CI-verified floors. Note the floors are enforced by the `min-versions` CI job.
- **Quickstart** — the `sweep → xarray.Dataset` example (`examples/sweep_to_dataset.py`),
  runnable, with the expected output shown.
- **Guides** — one task-focused how-to each:
  - *Workflows & sweeps*: `MultiRunMetricsWorkflow` and the Hydra multirun flow.
  - *Comparing methods*: `compare` for classification, the metric battery, the
    statistics (`test=`, Holm correction, the underpowered-test warning,
    single-seed behavior).
  - *Studies*: `Study` (train + compare in one call) and `Study.from_checkpoints`.
  - *Segmentation*: `task="segmentation"`, `num_classes`, `ignore_index` for void
    labels, **and the custom `predict_fn` recipe for models that return a dict**
    (e.g. torchvision seg models return `{"out": logits}`) — the real friction
    surfaced while dogfooding, so users don't hit it cold.
  - *Analyzing from Claude*: the MCP server (`mushin-mcp`, #32) — reuses and
    expands `docs/mcp.md`.
- **API Reference** — one markdown page per module, each containing
  `mkdocstrings` directives (e.g. `::: mushin.benchmark.compare`) that render the
  docstrings. Covers: `compare`, `BenchmarkResult`, `Study`, the `workflows`
  public API, `HydraDDP`, `MetricsCallback`, `load_experiment`,
  `load_from_checkpoint`.
- **Contributing** — points to / embeds `CONTRIBUTING.md` and the changelog/
  release process; links the `changes/` fragment workflow.

## Branding

Material theme with light/dark toggle; primary palette derived from the logo's
navy; `logos/mushin-dark.png` / `mushin-light.png` as the header/hero logo and
favicon. Code blocks use the Material syntax highlighting; copy-button enabled.

## Deployment

- New workflow `.github/workflows/docs.yml`: on push to `main` (and
  `workflow_dispatch`), build with `mkdocs build --strict` and deploy to
  **GitHub Pages** via `actions/deploy-pages`. Published at
  `https://martinez-hub.github.io/mushin/` (set `site_url` accordingly).
- One-time maintainer step (documented in RELEASING/CONTRIBUTING): enable Pages
  with **source = GitHub Actions** in repo settings.
- Single "latest" version — no `mike`/versioned docs yet (YAGNI at 0.x).

## Repo integration

- **`docs` CI gate**: a `docs` job in `ci.yml` runs `mkdocs build --strict` on
  pull requests, so broken links / bad `mkdocstrings` references fail CI. (Build
  only — deployment stays in `docs.yml` on `main`.)
- **Makefile**: `make docs` (`uv run mkdocs serve`) and `make docs-build`
  (`uv run mkdocs build --strict`).
- **README**: add a prominent link to the docs site near the top.
- **`changes/` fragment**: an `added` fragment for the docs site.
- Required-status-checks note: after first green run, the maintainer may add
  `docs` to branch protection (out of scope to change here; noted for follow-up).

## Data flow / build

`mkdocs build` reads `mkdocs.yml` → renders `docs/*.md` (Material) → resolves
`mkdocstrings` directives by importing `mushin` and reading its docstrings via
griffe → emits a static `site/`. The PR `docs` job builds with `--strict`
(warnings become errors). `docs.yml` on `main` builds the same way and uploads
`site/` to Pages.

## Error handling / quality

- `--strict` catches broken internal links, missing nav files, and unresolved
  `mkdocstrings` references at CI time.
- Because the reference renders live docstrings, the API docs cannot drift from
  the code (a renamed/removed symbol fails the strict build).
- mkdocstrings must import `mushin`, so the docs build installs the package +
  `docs` group (torch et al.); the job uses `uv sync --group docs`.

## Testing strategy

This is a docs/site project; "tests" are build-time verifications:
- `mkdocs build --strict` succeeds locally and in the PR `docs` job (no broken
  links, all `mkdocstrings` refs resolve).
- Every public symbol listed above renders on its reference page (spot-checked).
- The Quickstart example is the real `examples/` script (already covered by
  `tests/test_examples.py`), so its code stays correct.
- Nav has no orphan/!missing pages; `docs/superpowers/` is excluded from output.

## Non-goals

- Versioned docs (`mike`), custom domain, blog, or i18n.
- Rewriting docstrings (we render what exists; minor docstring fixes only if a
  symbol renders poorly).
- Changing branch protection (noted as a follow-up, not done here).
- Tutorials beyond the per-feature guides.
