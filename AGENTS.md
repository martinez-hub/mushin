# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Copilot, Cursor, and the like)
working in this repository. Humans should start with
[CONTRIBUTING.md](CONTRIBUTING.md); this file states the same rules, plus the
non-obvious ones that CI enforces and a newcomer would otherwise learn by
breaking them.

## What mushin is

A boilerplate-free sweep engine built on hydra-zen: decorate an experiment, sweep
its parameters, and get the results back as a labeled `xarray.Dataset`. The core
install is that sweep -> dataset path. An optional evaluation layer (`compare`,
the metric batteries, LLM evaluation, `Study`) lives behind the `eval` extra.
The PyPI distribution is `mushin-py`; the import name is `mushin`.

## Commands

Everything runs through [uv](https://docs.astral.sh/uv/). `make help` lists
the targets.

```bash
uv sync --extra eval      # dev environment (same as `make sync`)
make check                # lint + format-check + spell + tests: what CI runs
make lint                 # uv run ruff check .
make format               # uv run ruff format .
make spell                # codespell over the CI scope
make test                 # full suite, fast hypothesis profile
make docs-build           # mkdocs build --strict (CI fails on any warning)
```

Run one test file or test:

```bash
uv run --extra eval pytest tests/test_show.py -k mid_sweep \
    --hypothesis-profile fast -p no:cacheprovider
```

Execute the example notebooks the way CI does (only needed when you touch
`docs/notebooks/` or code they exercise):

```bash
uv run --extra eval --extra viz --extra netcdf \
    pytest --nbmake docs/notebooks/ -p no:cacheprovider
```

Tests marked `real_data` or `cluster` are deselected by default (they download
data or need a SLURM cluster). The `HydraDDP` tests skip without two GPUs. A
local pass therefore does not cover them, and you should not claim it does.

## Layout

| Path | What lives there |
| --- | --- |
| `src/mushin/__init__.py` | The public namespace, including the lazy-import tables (see below). |
| `src/mushin/workflows.py`, `_sweep_decorator.py` | `MultiRunMetricsWorkflow` and `@mushin.sweep`: the core. |
| `src/mushin/_resume.py`, `_sweep_io.py`, `_provenance.py` | Resumable sweeps, per-cell sidecars and manifest, provenance capture. |
| `src/mushin/_show.py`, `export.py` | Offline views of a sweep directory (`show`, `best`, `diff`, CSV export). |
| `src/mushin/_tuning.py`, `_packing.py` | Auto-tuning and GPU packing. |
| `src/mushin/benchmark/` | `compare`, the task registry, the batteries, the statistics (`eval` extra). |
| `src/mushin/llm/` | `compare_llms`, `compare_scores`, `llm_judge` (`eval` extra). |
| `src/mushin/study/` | `Study`: a training sweep routed into `compare` (`eval` extra). |
| `src/mushin/lightning/` | `HydraDDP`, `HydraFSDP`, `MetricsCallback`, cluster helpers. |
| `src/mushin/mcp/` | The read-only `mushin-mcp` server (`mcp` extra). |
| `tests/` | pytest + hypothesis. `tests/plugins/hydra_plugins/` holds a test sweeper plugin. |
| `examples/` | Runnable scripts; CI runs them through `tests/test_examples*.py`. |
| `docs/` | The mkdocs-material site. `docs/reference/` renders docstrings via mkdocstrings (`:::`). |
| `docs/notebooks/` | Example notebooks with committed outputs; CI re-executes them. |
| `docs/superpowers/` | Internal design specs and plans. Excluded from the site; never link to it from published docs. |
| `docs/llms.txt` | The LLM-facing index of the docs site, served at `/llms.txt`. |
| `changes/` | Towncrier news fragments (see below). |
| `paper/` | The paper source (LaTeX); built by `.github/workflows/paper-pdf.yml`. |

## Invariants that CI enforces

- **`import mushin` stays light.** A bare import must not load
  `pytorch_lightning`, `mushin.benchmark`, `mushin.llm`, `torchmetrics`, or
  `scipy`. Expose a name from a heavy subpackage by adding it to the
  `_LAZY_BENCHMARK` / `_LAZY_LIGHTNING` tables, the `TYPE_CHECKING` block, and
  `__all__` in `src/mushin/__init__.py`; never import it eagerly. Pinned by
  `tests/test_lazy_imports.py` and the `core-only` CI job.
- **The core works with no extras.** Code under `benchmark/`, `llm/` and
  `study/` may use `torchmetrics`/`scipy`; nothing else may. Touching an `eval`
  name without the extra must raise an `ImportError` that names
  `mushin-py[eval]`.
- **Dependency floors are justified and tested.** Every floor in
  `pyproject.toml` carries a comment saying why it is that version, and the
  `min-versions` job runs the suite at the floors (`make test-lowest`
  reproduces it in Docker). Add a comment with any new floor, and do not add
  upper bounds on library dependencies.
- **`uv.lock` is generated.** Change dependencies in `pyproject.toml`, then run
  `uv lock` (or `uv sync`). Never edit the lockfile by hand.
- **Python 3.11 is the floor** (`target-version = "py311"`); 3.11 to 3.14 are
  the supported, CI-required versions.
- **Documented numbers can be pinned by tests.** `tests/test_docs_claims.py`
  checks that thresholds quoted in `docs/guides/statistics.md` match the code.
  If you change one side, change the other; do not just edit the test.
- **`docs/llms.txt` lists every nav page.** Adding, renaming or removing a page
  in `mkdocs.yml`'s `nav` means editing `docs/llms.txt` too;
  `tests/test_llms_txt.py` fails otherwise.
- **Notebooks declare their own installs.** Each notebook shows a
  `%pip install ...` line, and `scripts/check_notebook_prereqs.py` runs it in a
  venv holding exactly that, so the line must cover every import. Outputs are
  committed; the `nbstripout` hook runs with `--keep-output --keep-id`. Execute
  a notebook against a clean, committed tree so its provenance records
  `dirty=false`.

## Conventions

- **Every PR adds a news fragment** under `changes/`, named
  `<PR-number>.<type>.md` with `<type>` one of `added`, `changed`, `fixed`,
  `removed`, `deprecated`, `misc`. Before the PR exists, use `+<slug>.<type>.md`
  and rename it to the number once it does. A `misc` fragment (CI, refactor,
  typo; nothing user-facing) is an empty file. See `changes/README.md`.
- **Never edit `CHANGELOG.md` directly.** CI rejects it unless the PR has the
  `changelog-exempt` label, which only release PRs use (`RELEASING.md`).
- **Commit messages** use conventional prefixes, as in `git log`: `feat:`,
  `fix:`, `docs:`, `test:`, `ci:`, `build:`, `build(deps):`, `chore:`,
  `release:`. The subject says what changed; the body says why.
- **Public API** is the top-level `mushin` namespace plus the documented
  `mushin.benchmark` and `mushin.llm` symbols. Underscore modules are internal.
  The project is pre-1.0: a breaking change goes in a minor release with
  migration notes in its fragment, and a moved top-level name keeps working for
  a release through the `_DEPRECATED` table in `src/mushin/__init__.py`.
- **Docstrings** on public functions are NumPy style; mkdocstrings renders
  them into the API reference, so they are user-facing docs.
- **Comments explain why.** The codebase documents the reason behind each
  floor, workaround and guard (read `pyproject.toml` for the register). Match
  that density in code you touch.
- **License header.** Most source and test files open with the MIT header:
  `# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY` then
  `# SPDX-License-Identifier: MIT`. Copy it from a neighboring file.
- **Docs prose avoids em dashes** (#229). Use a colon, semicolon, comma or
  parentheses, whichever fits the clause. `CHANGELOG.md` is shipped history
  and keeps its original punctuation.
- **Python inside Markdown is formatted too.** `ruff format --check .` covers
  code blocks in `.md` files and cells in `.ipynb` files, so a doc snippet can
  fail CI.
- **Tests** that run a sweep write output directories: use `tmp_path` or the
  `cleandir` fixture. A test that stores a ConfigStore group marks itself
  `@pytest.mark.config_group("name")` so the group is removed afterwards. The
  benchmark task registry is restored after every test automatically.

## Before you push

1. `make check` passes.
2. Docs touched (including `mkdocs.yml` or `docs/llms.txt`): `make docs-build`
   passes.
3. Notebooks or code they exercise touched: the nbmake command above passes.
4. A fragment exists in `changes/`.

## Using mushin from an agent

If you are writing code that *uses* mushin rather than changing it, read the
docs index at <https://martinez-hub.github.io/mushin/llms.txt>. To inspect
completed runs, the `mushin-mcp` server (`pip install "mushin-py[mcp]"`) gives
an MCP client read-only tools over a directory of runs; see
[docs/guides/mcp.md](docs/guides/mcp.md).
