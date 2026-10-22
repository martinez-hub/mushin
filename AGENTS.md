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
| `src/mushin/agent_guide.md`, `__main__.py` | The usage guide for agents that *use* mushin, printed by `python -m mushin guide` and shown at `docs/for-agents.md`. |
| `src/mushin/workflows.py`, `_sweep_decorator.py` | `MultiRunMetricsWorkflow` and `@mushin.sweep`: the core. |
| `src/mushin/_resume.py`, `_sweep_io.py`, `_provenance.py` | Resumable sweeps, per-cell sidecars and manifest, provenance capture. |
| `src/mushin/_show.py`, `export.py` | Offline views of a sweep directory (`show`, `best`, `diff`, CSV export). |
| `src/mushin/_tuning.py`, `_packing.py` | Auto-tuning and GPU packing. |
| `src/mushin/benchmark/` | `compare`, the task registry, the batteries, the statistics (`eval` extra). |
| `src/mushin/llm/` | `compare_llms`, `compare_scores`, `llm_judge` (`eval` extra). |
| `src/mushin/study/` | `Study`: a training sweep routed into `compare`. Its features need the `eval` extra, but `import mushin` loads it. |
| `src/mushin/lightning/` | `HydraDDP`, `HydraFSDP`, `MetricsCallback`, cluster helpers. |
| `src/mushin/mcp/` | The read-only `mushin-mcp` server (`mcp` extra). |
| `tests/` | pytest + hypothesis. `tests/plugins/hydra_plugins/` holds a test sweeper plugin. |
| `examples/` | Runnable scripts; CI runs them through `tests/test_examples*.py`. |
| `docs/` | The mkdocs-material site. `docs/reference/` renders docstrings via mkdocstrings (`:::`). |
| `docs/notebooks/` | Example notebooks with committed outputs; CI re-executes them. |
| `docs/superpowers/` | Internal design specs and plans. Excluded from the site; never link to it from published docs. |
| `docs/llms.txt` | The LLM-facing index of the docs site, served at <https://martinez-hub.github.io/mushin/llms.txt>. |
| `changes/` | Towncrier news fragments (see below). |
| `paper/` | The paper source (LaTeX); built by `.github/workflows/paper-pdf.yml`. |

## Invariants that CI enforces

- **`import mushin` stays light.** A bare import must not load `torch`,
  `pytorch_lightning`, `torchmetrics`, `scipy`, `mushin.benchmark` or
  `mushin.llm` (`tests/test_lazy_imports.py` checks each one). Expose a name
  from a heavy subpackage by adding it to the `_LAZY_BENCHMARK` /
  `_LAZY_LIGHTNING` tables, the `TYPE_CHECKING` block, and `__all__` in
  `src/mushin/__init__.py`; never import it eagerly.
- **The core works with no extras.** Only `benchmark/` and `llm/` may import
  `torchmetrics`/`scipy` at module level, because `import mushin` loads them
  lazily. scipy is absent from a core install, so a stray eager import breaks
  it; torchmetrics arrives anyway through `pytorch-lightning`, so a stray
  import of it costs import time instead. `study/` is imported eagerly (`from .study import Study`), so it
  imports `mushin.benchmark` inside functions or under `TYPE_CHECKING`, as
  `study/_study.py` and `study/_load.py` do. Touching an `eval` name without
  the extra must raise an `ImportError` that names `mushin-py[eval]`; the
  `core-only` CI job installs no extras and checks that.
- **Dependency floors are justified and tested.** Most floors in
  `pyproject.toml` carry a comment saying why they are that version
  (`omegaconf`, `torchmetrics`, `netCDF4`, `lpips`, `pystoi` and `mcp` do not
  yet); add one with any new floor. The `min-versions` job runs the suite with
  the core, the `eval` extra and the dev group at their floors
  (`make test-lowest` reproduces it in Docker). It does not install the
  `detection`, `image` or `audio` extras, so those floors are not floor-tested.
  Avoid upper bounds: a ceiling on a library propagates into every downstream
  resolution. Three caps exist. `mcp >= 2, < 3` dates from the move to the 2.x
  SDK, which removed the 1.x module the server was built on (0.12.0 in
  `CHANGELOG.md`, #163); the cap guards against the next major doing the same,
  though no comment says so. The docs group's `mkdocs-material < 10` and
  `mkdocstrings[python] < 2` constrain only the docs build. Ask the maintainer
  before adding or removing a cap.
- **Python 3.11 is the floor** (`target-version = "py311"`); 3.11 to 3.14 are
  the supported, CI-required versions.
- **Documented numbers can be pinned by tests.** `tests/test_docs_claims.py`
  checks that thresholds quoted in `docs/guides/statistics.md` match the code.
  If you change one side, change the other; do not just edit the test.
- **`docs/llms.txt` lists every nav page.** Adding, renaming or removing a page
  in `mkdocs.yml`'s `nav` means editing `docs/llms.txt` too.
  `tests/test_llms_txt.py` fails otherwise; it also checks that the install
  name, Python range and extras listed there match `pyproject.toml`, and runs
  its code example. CI runs it, `tests/test_docs_claims.py` and
  `tests/test_agent_guide.py` in the `docs` job on every PR, since the `test`
  job skips docs-only PRs.
- **The agent guide is tested.** `src/mushin/agent_guide.md` is ASCII, its
  python blocks run in order as one script, the keywords of its `run()` calls
  exist and its links are nav pages (`tests/test_agent_guide.py`); its seed
  floors are pinned in `tests/test_docs_claims.py`. When you change behavior
  the guide describes, change the guide.
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
- **No AI attribution.** Commits, PR bodies, code and docs carry no
  `Co-Authored-By:` trailers, "Generated with" footers or session links for AI
  tools.
- **`uv.lock` is generated.** Change dependencies in `pyproject.toml`, then run
  `uv lock` (or `uv sync`) and commit both. Never edit the lockfile by hand. CI
  does not check that the lock is current: its `uv sync` re-resolves a stale
  lock without failing.
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
- **Tests** that run a sweep write a `multirun/` directory into the current
  directory unless told otherwise. Pass `working_dir=tmp_path` to `run(...)`,
  as the existing tests do, or request the `cleandir` fixture, which changes
  into a temporary directory. Requesting `tmp_path` alone isolates nothing.
- A test that stores a ConfigStore group must both mark itself
  `@pytest.mark.config_group("name")` and request the `restore_config_group`
  fixture; the fixture is not autouse, so the marker alone does nothing (see
  `tests/test_workflows.py`). The benchmark task registry, by contrast, is
  restored after every test automatically.

## Before you push

1. `make check` passes.
2. Docs or `src/mushin/agent_guide.md` touched (including `mkdocs.yml` or
   `docs/llms.txt`): `make docs-build` passes, and so does
   `uv run --extra eval pytest tests/test_llms_txt.py tests/test_docs_claims.py tests/test_agent_guide.py`.
3. Notebooks or code they exercise touched: the nbmake command above passes.
4. A fragment exists in `changes/`.

## Using mushin from an agent

If you are writing code that *uses* mushin rather than changing it, run
`python -m mushin guide` (mushin 0.16 or later), or read the docs index at
<https://martinez-hub.github.io/mushin/llms.txt>. To inspect completed runs, the
`mushin-mcp` server (`pip install "mushin-py[mcp]"`) gives an MCP client
read-only tools over a directory of runs; see
[docs/guides/mcp.md](docs/guides/mcp.md).
