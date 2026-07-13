# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- towncrier release notes start -->

## [0.4.0] - 2026-07-13

### Added

- Four new built-in task batteries — `regression`, `image_quality`, `audio`, and
  `retrieval` — plus a per-`Task` `update_fn` hook for metrics whose update step is
  not `(preds, target)` (used by `retrieval`). LPIPS and STOI sit behind the
  optional `[image]` and `[audio]` extras. Each battery is exported from `mushin`.
- Overhauled the documentation: runnable, tested example scripts (MNIST) that the guides embed verbatim, deeper guides with annotated output, and new Tutorial, Core concepts, Custom metrics/predict_fn, and Statistics pages.
- Public task API: `Task` dataclass plus `register_task`, `get_task`, and
  `list_tasks` make evaluation tasks first-class and reusable. `compare(...)` and
  `Study(...)` now accept either a `Task` object or a registered task name, and the
  built-in batteries (`classification_battery`, `segmentation_battery`,
  `detection_battery`) are exported from `mushin`.
- `compare(task="detection")` — compare trained object detectors across seeds over
  the full `torchmetrics.detection` bounding-box family (mean-average-precision plus
  the IoU/GIoU/CIoU/DIoU variants), reporting every scalar metric with Holm-corrected
  significance. Needs the optional `mushin-py[detection]` extra.
- `mushin.llm.compare_llms` — compare LLM systems (callables or hydra-zen configs) across reproducible stochastic seeds with a metric (a plain scorer, a `torchmetrics` text metric, or the new `llm_judge` helper), reporting Holm-corrected statistical significance. Includes an on-disk output cache. Provider-agnostic: you bring the systems, data, and judge model.
- `tune_batch_size` / `tune_learning_rate`: opt-in, reproducibility-preserving
  auto-tuning. Lightning's batch/LR finder runs once, the result is pinned to a
  sidecar YAML, and later runs reuse it. `tune_batch_size` pins a hardware-
  independent effective batch, choosing the largest device batch that both fits and
  divides the per-device target exactly, so the effective batch is identical on any
  GPU count with no drift.

### Changed

- Modernized the codebase to Python 3.10+ idioms now that 3.9 is no longer
  supported: `ruff` `target-version` is `py310`, and the pyupgrade auto-fixes
  (`Optional[X]`/`Union[X, Y]` -> `X | None`/`X | Y`) plus explicit `zip(..., strict=True)`
  have been applied. No behavior change.
- `import mushin` is now lightweight: the `benchmark` and `llm` subsystems load on
  first use instead of at import time, so a bare import no longer pulls the
  battery/eval machinery. Every existing top-level name still resolves. The default
  Hydra config/job name is now `mushin_workflow` (was `rai_workflow`), and the new
  `mushin.original_cwd()` helper anchors relative paths in `task()` against the
  launch directory rather than Hydra's per-job output directory.

### Fixed

- Fixed the docs example scripts surfaced in review: the Study example now trains and evaluates on separate MNIST splits and resolves its checkpoint directory to an absolute path (so it works under Hydra's per-job chdir); the custom-metrics guide no longer implies `Study` accepts `metrics`/`predict_fn` (those are `compare`-only).
- Hardening from a repo-wide adversarial audit: `load_experiment` now loads DDP/nested-layout configs (was silently `None`) and labels each job's own `working_dir`; the benchmark `compare()` path masks zero within-group-variance comparisons (no more false-positive significance — now consistent with `compare_llms`, both via a single `compare_methods`); `MetricsCallback` keeps every metric series aligned to the epoch axis (NaN-padding missing metrics, reserving the `epoch` key); `Study` labels the seed coordinate with the real seed values and relocates checkpoints across filesystems (`shutil.move`); multirun overrides are built with Hydra's `choice(...)` syntax so comma-bearing and single-element values are no longer mis-split; and several legibility/robustness fixes (`to_dataset` empty-method error, `_to_device` namedtuple support, `'='`-bearing override parsing, `load_from_dir` cache reset, dropped the never-working dict-override path).

### Removed

- Dropped support for Python 3.9 (end-of-life October 2025); `mushin` now requires
  Python >= 3.10. This refreshes the dependency lockfile to patched versions of
  pillow, urllib3, aiohttp, filelock, requests, pytest, and pytorch-lightning,
  clearing the Dependabot security alerts anchored on the old Python-3.9 dependency
  branch. The `scipy` (>= 1.13) and `matplotlib` (>= 3.9) floors are raised to their
  first NumPy-2-compatible releases, and the `mcp` extra no longer needs a Python
  version gate.

### Deprecated

- `BaseWorkflow` and `RobustnessCurve` are deprecated at the top level and will be
  removed in a future release. Import them from `mushin.workflows` instead;
  accessing them as `mushin.BaseWorkflow` / `mushin.RobustnessCurve` now emits a
  `DeprecationWarning`.

### Misc

- Added a `@claude` mention bot (GitHub Actions): mention `@claude` in any issue or PR comment to have Claude answer questions or make changes. PR *reviews* stay with the Codex connector.


## [0.3.0] - 2026-06-24

### Added

- Optional read-only MCP server (`mushin-mcp`, `pip install "mushin-py[mcp]"`, Python >= 3.10) that lets Claude Code and other MCP clients list experiments, summarize swept parameters and metrics, read configs, and inspect saved datasets — with no training or sweep launching. (#32)
- Added a documentation website (MkDocs Material) with how-to guides and an auto-generated API reference, deployed to GitHub Pages.
- The test suite now runs on Windows in CI (windows-latest) across Python 3.9-3.14, alongside Linux.

### Changed

- Raised minimum dependency floors to a tested minimum and added a `min-versions` CI job that runs the suite against the lowest declared versions, so the floors stay honest. The old floors did not actually work: several were incompatible with NumPy 2 (the non-Intel floor), so `torch` (>= 2.4), `pandas` (>= 2.2.2), `xarray` (>= 2024.6), and `netCDF4` (>= 1.7.0) were raised to their first NumPy-2-compatible releases. `pytorch-lightning` was raised to >= 2.4 (the pre-2.0 1.5 API does not run this package) and `hydra-zen` to >= 0.10 (for `multirun`). Intel-macOS keeps torch 2.2.x / NumPy 1.x.


## [0.2.1] - 2026-06-23

### Added

- Adopted towncrier news fragments with a CI gate — each PR now adds a `changes/` fragment, assembled into the changelog at release.

### Fixed

- Point PyPI project links at the mushin repo (Homepage/Repository/Issues/Changelog) instead of only the upstream toolbox.
- Synced `uv.lock`'s recorded project version to 0.2.0.

## [0.2.0] - 2026-06-23

### Added
- `Study` — orchestrate a multi-seed training sweep (via Hydra/Lightning) and
  route the trained models into `compare` in one call;
  `Study.from_checkpoints(...)` for eval-only comparison of existing checkpoints.
- Segmentation support in `compare` (`task="segmentation"`): mean IoU, Dice,
  pixel accuracy, and macro precision/recall, with `ignore_index` for void
  labels; plumbed through `Study`.
- Streaming evaluation — metrics update per batch (O(C²) memory for the
  segmentation battery); one unified eval loop for all tasks.
- A task registry so new task types self-describe their battery and predict step.
- `compare` gains a `prob_metrics` override and rejects one-shot `data` iterators.

### Changed
- Package author/maintainer set to Josue Martinez-Martinez (previously the
  original MIT-LL authors; their attribution remains in LICENSE and the README).
- README: documented `compare` and `Study`; clarified install
  (`pip install mushin-py` → `import mushin`); absolute logo URLs so images
  render on PyPI; transparent dark-mode logo.

### Fixed
- Significance: `compare` warns when the chosen test cannot reach `alpha` at the
  given seed count (e.g. Wilcoxon with ≤5 seeds).
- `cohens_d` returns signed infinity (not `0.0`) when within-group variance is
  zero but the means differ.
- Several review-driven fixes: Hydra-scalar method names in `Study` sweeps,
  NaN-safe Holm correction, ragged-results validation, and an empty-checkpoints
  guard.

## [0.1.0] - 2026-06-22

First release of `mushin` as a standalone package — a fork of the
`rai_toolbox.mushin` workflow layer from MIT Lincoln Laboratory's
(unmaintained) responsible-ai-toolbox.

### Added
- Standalone packaging: `pyproject.toml`, MIT license (original MIT-LL copyright
  retained), README, and a vendored `value_check` so the package no longer
  depends on `rai_toolbox`.
- uv-based development workflow with a committed `uv.lock`.
- Ruff (lint + format) and codespell, with pre-commit hooks.
- `Makefile` developer shortcuts (`make check`, `make test-py`, ...).
- GitHub Actions CI across Python 3.9–3.14.
- Community health files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue forms,
  PR template, Dependabot, CODEOWNERS.

### Changed
- Modernized type hints (PEP 585 generics, `collections.abc` imports).
- Declared support for Python 3.9–3.14; NumPy >= 2 except on Intel-macOS, which
  is capped at NumPy 1.x / torch 2.2.x (no newer wheels on that platform).

### Fixed
- Compatibility with PyTorch 2.6+, which changed `torch.load`'s default
  `weights_only` to `True`: pass `weights_only=False` when loading trusted,
  self-produced metric and checkpoint files.
- `test_overrides_roundtrip`: exclude Hydra-reserved tokens (`null`/`none`/
  `nan`/`inf`) from the generated-string strategy.
- Updated deprecated `xarray.Dataset.dims` to `.sizes` in tests.

[Unreleased]: https://github.com/martinez-hub/mushin/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/martinez-hub/mushin/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/martinez-hub/mushin/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/martinez-hub/mushin/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/martinez-hub/mushin/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martinez-hub/mushin/releases/tag/v0.1.0
