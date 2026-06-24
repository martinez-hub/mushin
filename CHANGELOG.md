# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- towncrier release notes start -->

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

[Unreleased]: https://github.com/martinez-hub/mushin/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/martinez-hub/mushin/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/martinez-hub/mushin/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/martinez-hub/mushin/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martinez-hub/mushin/releases/tag/v0.1.0
