# Contributing to mushin

Thanks for your interest in contributing! mushin is a community-maintained
fork of the `rai_toolbox.mushin` workflow layer. Contributions of all kinds are
welcome: bug reports, fixes, docs, and features.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/martinez-hub/mushin
cd mushin
uv sync                 # create the dev environment from uv.lock
pre-commit install      # optional but recommended (see below)
```

## The local gate

Before opening a PR, please run the same checks CI runs:

```bash
make check              # lint + format-check + spell + tests
```

Or individually:

```bash
make lint               # ruff check
make format             # ruff format
make spell              # codespell
make test               # pytest (fast hypothesis profile)
make test-py PYTHON=3.12 # run the suite on a specific Python version
```

We also provide [pre-commit](https://pre-commit.com/) hooks that run ruff and
codespell on changed files automatically:

```bash
pre-commit install      # then hooks run on every `git commit`
pre-commit run --all-files
```

### A note on torch versions

PyTorch's behavior changes across versions (e.g. `torch.load` defaults). If your
change touches model/metric loading or anything torch/numpy-version sensitive,
please verify against a recent torch — the CI matrix (Python 3.9–3.14) is the
source of truth. Contributors on Intel macOS cannot install torch > 2.2; a Linux
container or CI is the reliable way to test newer torch there.

## Submitting changes

1. Fork the repo and create a topic branch from `main`.
2. Make your change, with tests. We use `pytest` + `hypothesis`.
3. Ensure `make check` passes.
4. Add a news fragment under `changes/` (see [Changelog](#changelog) below).
5. Open a pull request against `main`, filling out the PR template and linking
   any related issue.
6. CI (lint, the Python test matrix, and the changelog gate) must pass. `main`
   is protected and merges require a passing CI run.

### Changelog

Every PR with a user-facing change must add a **news fragment** under
[`changes/`](changes/README.md): a file named `changes/<PR-or-issue>.<type>.md`
(types: `added`, `changed`, `fixed`, `removed`, `deprecated`). One line of
Markdown, written for users. For example, `changes/123.fixed.md` containing:

```
`Study.from_checkpoints` now validates that the checkpoint list is non-empty.
```

CI enforces this. For CI-only, refactor, or typo PRs with nothing user-facing,
add a `misc` fragment instead (e.g. `changes/42.misc.md`) — it satisfies the gate
and adds only a terse `- #42` under a "Misc" section. See
[`changes/README.md`](changes/README.md) for details.

## Style

- Formatting and linting are handled by **ruff** (config in `pyproject.toml`).
- Public functions should keep their NumPy-style docstrings.
- Add or update tests for any behavior change.

## Reporting bugs / requesting features

Use the issue templates (bug report / feature request). For **security
vulnerabilities**, do not open a public issue — see [SECURITY.md](SECURITY.md).
