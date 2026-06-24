# Contributing

Thanks for your interest in contributing! This page summarizes the key steps —
see [CONTRIBUTING.md](https://github.com/martinez-hub/mushin/blob/main/CONTRIBUTING.md)
on GitHub for the full details.

## Dev setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/martinez-hub/mushin
cd mushin
uv sync            # create the dev environment from uv.lock
pre-commit install  # recommended: runs ruff + codespell on each commit
```

## The local gate

Before opening a PR, run the same checks CI runs:

```bash
make check   # lint + format-check + spell + tests
```

Individual shortcuts: `make lint`, `make format`, `make spell`, `make test`,
`make test-py PYTHON=3.12`.

## Changelog fragments

mushin uses [towncrier](https://towncrier.readthedocs.io/) for changelogs.
Every user-facing PR needs a news fragment in `changes/`:

```
changes/<id>.<type>.md
```

Where `<id>` is the GitHub issue or PR number (or a short slug for pre-issue
work), and `<type>` is one of `added`, `changed`, `fixed`, `removed`, or
`deprecated`. The fragment body is a single plain-English sentence describing
the change.

Example:
```
changes/42.added.md
```
```
Added `ignore_index` support to `compare` for segmentation tasks.
```

The CI `changelog` job will fail if a PR omits a fragment (unless the PR has
the `changelog-exempt` label, reserved for release PRs).

## Releases

See [RELEASING.md](https://github.com/martinez-hub/mushin/blob/main/RELEASING.md)
on GitHub for the release checklist.
