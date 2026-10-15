# Design: a usage guide for AI coding agents

**Date:** 2026-10-15
**Status:** Draft, before spec review. Decisions were made without asking the
user, at their request; each records its rationale and the alternatives it
rejected.
**Sequencing:** lands after the docs snippet-include fix (#249). D4 relies on
that PR's include guard, so this branch is rebased onto it before the PR.

## Problem

Scientists and engineers increasingly ask a coding agent (Claude Code, Codex,
Cursor and the like) to set up their experiments. mushin is a niche library
that agents know little about from training. Nothing in the repo is written for
them:

- There is no agent-oriented page, no `AGENTS.md`, no `llms.txt`, and the docs
  build emits nothing machine-readable beyond HTML.
- `import mushin; help(mushin)` shows no module docstring, and there is no
  `python -m mushin`. An agent that installed the package has no
  version-matched entry point.
- What an agent needs is spread across the quickstart, concepts, workflows,
  exploration-to-paper, resilience and statistics pages. No page says which
  API to use for which job, shows one runnable end-to-end template, lists the
  on-disk output layout, or says what to check before reporting results.
- Pitfalls that produce wrong results or wasted compute are not collected in
  one place. Examples verified on 2026-10-15:
  - `train.run(lr=[0.1, 0.2])` raises: swept values must be wrapped in
    `mushin.multirun([...])`. The error says so, but only after the agent
    guessed wrong.
  - `mushin.best(root, "acc")` returns the single best *cell*, which is one
    lucky seed, not the best configuration averaged over seeds.
  - `mushin.show(root)` prints its table and also returns it, so
    `print(mushin.show(root))` prints it twice.
  - `run()`'s docstring documents `dry_run`, `confirm_above`,
    `max_total_seconds`, `on_error`, `resume`, `sample` and `cache_dir`, but
    not `working_dir` or `launcher`.
- The docs' example includes were broken from 2026-07-27 to 2026-10-15 (ruff
  0.16 respaced them; fixed separately). An agent reading the Tutorial in that
  window saw a directive instead of code. Agent-facing code must be tested, not
  just rendered.

## Goal

An agent that has installed mushin, or can fetch its docs, can set up a
correct sweep, run it safely, analyze it, and report it, without reading the
source. Success criteria:

1. One guide, reachable offline from the installed package and online from the
   docs site, `llms.txt`, the README and `help(mushin)`.
2. The guide's Python code runs as written. CI executes it, so API drift fails
   a test instead of misleading agents.
3. The guide matches the installed version, because it ships inside the
   package.

## Non-goals

- Changing mushin's behavior. The `best()` and `show()` quirks are documented,
  not changed. Changing them is a separate decision.
- MCP changes. The server stays analysis-only, and the guide points to it.
- Packaging the guide as a Claude Code skill or other vendor-specific format.
- An `AGENTS.md` for agents that *contribute* to mushin. This guide is for
  agents *using* mushin in someone else's project.
- Filling the API reference gaps (exports with no reference page). The guide
  tells agents to read docstrings with `help()`, which always matches the
  installed version.

## Facts established (2026-10-15, mushin 0.15.1)

- **Sweep output layout:** each sweep writes `multirun/<date>/<time>/` under
  the launch directory unless `working_dir=` is given. It holds
  `multirun.yaml`, `mushin_sweep_manifest.json` and one directory per cell
  (`0/`, `1/`, ...). Each cell directory holds `mushin_metrics.json`,
  `mushin_cell_status.json`, `mushin_provenance.json`, `mushin_workflow.log`
  and `.hydra/`. `sweep.workflow.working_dir` is the sweep root.
- **`dry_run=True`** prints the cell count and every swept axis and returns
  `{"num_cells", "axes", "fixed", "working_dir"}` without launching a job.
- **Statistics without torch:** a sweep over `method` and `seed` returns a
  `(method, seed)` dataset. `mushin.benchmark.compare_methods(ds)` returns the
  pairwise table (Wilcoxon by default, Holm-corrected). Verified with 2
  methods × 6 seeds. It needs the `eval` extra (scipy).
- **Reloading:** `MultiRunMetricsWorkflow().load_from_dir(path, metrics_filename)`
  reloads a finished sweep. `mushin.show`, `mushin.best` and `mushin.diff`
  read a sweep root directly.
- **Packaging:** `[tool.setuptools.package-data]` ships only `py.typed` today.
  The `batteries-clean-install` CI job builds the wheel and installs it into a
  clean venv.
- **`mkdocs-llmstxt` 0.5.1** (maintained by the mkdocstrings author) generates
  `/llms.txt` and `/llms-full.txt` from selected pages. It requires Python >=
  3.11, which the docs build already uses.

## Approaches considered

1. **Docs-site page plus `llms.txt` only.** Least work, but agents without web
   access can't reach it, and the site describes the latest release, not the
   installed one.
2. **A guide shipped in the package, surfaced everywhere (recommended).** One
   Markdown file inside `mushin/`, printed by `python -m mushin guide`,
   included verbatim in a docs page, listed first in `llms.txt`, and pointed
   to by the README and the module docstring. CI executes its code blocks and
   checks the built wheel contains it.
3. **Approach 2 plus agent integrations** (a skill package, MCP prompts or
   resources, setup tools in the MCP server). More reach for specific agents,
   but more surface to maintain, and vendor formats are still moving. It can be
   built on approach 2 later, since the guide is the content either way.

## Decisions

### D1. One guide file, shipped in the package

`src/mushin/agent_guide.md`, added to `[tool.setuptools.package-data]`.

**Why:**
- An agent with no network still gets it.
- It matches the installed version.
- A file in the package is discoverable by an agent browsing
  `site-packages/mushin/`.

**Name:** not `AGENTS.md`. Tools such as Codex treat an `AGENTS.md` in a
directory as instructions for working on that directory's code, so an agent
editing mushin's own `src/mushin/` would take user-facing guidance as
contributor instructions.

### D2. `python -m mushin guide` prints it

Add `src/mushin/__main__.py` with an `argparse` CLI:
- `python -m mushin guide` prints the guide to stdout, preceded by one line,
  `<!-- mushin X.Y.Z -->`, with the installed version from
  `importlib.metadata.version("mushin-py")`. An HTML comment keeps the output
  valid Markdown and invisible when rendered.
- `python -m mushin guide --path` prints the file's path, for agents that
  read files with their own tools.
- `python -m mushin` with no command prints help and exits 0.

**Why `python -m` and not a console script:** it works in any environment
where mushin is importable, including notebooks and `uv run`, and needs no
`[project.scripts]` entry. It is also what an agent tries first.

**Rejected:** a `mushin.guide()` function. It is less discoverable than the
CLI, and the module docstring covers interactive discovery (D3).

### D3. A module docstring for `help(mushin)`

Add a short module docstring to `src/mushin/__init__.py`. It gives a
one-paragraph description, the three entry points (`@mushin.sweep`,
`MultiRunMetricsWorkflow`, `Study` / `compare`), and the line "AI coding
agents: run `python -m mushin guide` for a usage guide matching this
version".

### D4. A docs page that includes the guide verbatim

`docs/agents.md` ("For AI agents"), in the top-level nav after Quickstart:
- A short intro for the human: how to point an agent at the guide, with a
  copy-paste block for the project's `AGENTS.md` / `CLAUDE.md`, plus the MCP
  server for analysis.
- Then the guide, included with a one-line snippet include outside any code
  fence (`--8<-- "src/mushin/agent_guide.md"`), the same form as
  `docs/changelog.md`. The docs' include guard allows this form outside
  fences.

**Links inside the guide** are absolute docs-site URLs
(`https://martinez-hub.github.io/mushin/...`), so they work in the terminal,
in the package file and on the site. A test checks that each one maps to an
existing docs page (D7).

### D5. `llms.txt` via mkdocs-llmstxt

Add `mkdocs-llmstxt` to the `docs` dependency group and configure it:
- The "Start here" section lists the agents page first, then Quickstart and
  Core concepts.
- A Guides section follows.
- Generate `llms-full.txt` too.

**Fallback:** if the plugin cannot pass `mkdocs build --strict` alongside
mkdocs-jupyter and mkdocstrings, ship a hand-written `docs/llms.txt` index (no
plugin; MkDocs copies it to the site root) and drop `llms-full.txt`. The plan
verifies this first.

### D6. README pointer

A short "Using mushin with an AI agent" section right after the README
Quickstart. It covers `python -m mushin guide`, the docs page, `llms.txt`,
and the copy-paste project-instructions block. The README is also the PyPI
description, which is the first page an agent fetches.

### D7. Tests keep the guide honest

A new test file, `tests/test_agent_guide.py`:
1. **The guide ships:** `importlib.resources.files("mushin") / "agent_guide.md"`
   is a file.
2. **Its code runs:** execute every ```python block of the guide in order, in
   one namespace, inside a temporary working directory. The guide therefore
   uses only runnable blocks for code. Shell commands go in ```bash blocks and
   illustrative output in ```text blocks, neither of which is executed. The
   test needs the `eval` extra (it skips without scipy), and the CI `test` job
   has it.
3. **Its links resolve:** every `https://martinez-hub.github.io/mushin/<path>`
   URL in the guide maps to `docs/<path>.md` or `docs/<path>/index.md`
   (anchors stripped).
4. **The CLI works:** `main(["guide"])` prints the version line and then the
   guide, `main(["guide", "--path"])` prints an existing path, and `main([])`
   exits 0.
5. **Its numbers are pinned:** the seed floor the guide states for the
   Wilcoxon test (6 at alpha 0.05) is checked against
   `mushin.benchmark._stats.warn_if_underpowered`, the way
   `tests/test_docs_claims.py` checks the statistics guide.

In CI, the `batteries-clean-install` job (built wheel, clean venv) also runs
`python -m mushin guide --path` and checks the file is non-empty. That proves
the wheel carries it, which an editable install cannot.

### D8. Guide content

Compact: about 250 to 350 lines, because a long guide gets skimmed. In order:

1. **What mushin is.** Sweep a Python function over a grid of parameters and
   seeds, and get a labeled `xarray.Dataset` / `pandas.DataFrame` back, with
   provenance and paired statistics. Framework-agnostic sweeps; torch and
   Lightning for the eval layer.
2. **Install.** The extras that matter (`eval`, `viz`, `netcdf`, the batteries,
   `mcp`) in a small table.
3. **Choosing an entry point.** A decision table:
   - a function over a grid: `@mushin.sweep`
   - custom collection or Lightning: `MultiRunMetricsWorkflow`
   - (method × seed) significance: `mushin.benchmark.compare_methods`
   - torch models on a battery: `mushin.compare`
   - train then compare: `Study`
   - LLMs: `mushin.llm`
   - analyzing finished runs: `show` / `best` / `diff` / MCP
   - parallel or cluster runs: launchers
4. **The template (runnable, tested).** Define a task that returns a dict of
   metrics. Preview with `dry_run=True`. Run with `method` and `seed` axes and
   a `working_dir`. Read the results as a dataset and a DataFrame. Run
   `compare_methods`. Show where the outputs went.
5. **Rules that prevent wrong results or wasted compute:**
   - `multirun([...])` for swept values; `hydra_list([...])` for a list-valued
     argument.
   - Return a dict of numbers. The values are JSON-coerced into
     `mushin_metrics.json`.
   - Each cell runs in its own directory: use `mushin.original_cwd()` for
     input paths.
   - `dry_run=True` before anything expensive, plus `confirm_above` /
     `MUSHIN_MAX_CELLS` and `max_total_seconds`.
   - `on_error="nan"` keeps sweeping, and its failures are listed.
     `resume=True` with the same `working_dir` re-runs only incomplete cells.
     `sample=` explores a subset.
   - Enough seeds for the test: Wilcoxon cannot reach alpha 0.05 below 6 seeds
     (pinned by D7 item 5).
   - Commit code before final runs: provenance records the git SHA and whether
     the tree was dirty.
   - `best()` is per-cell. Average over seeds to pick a configuration.
     `show()` prints and returns.
6. **Where results go and how to reload them.** The layout, `load_from_dir`,
   `show` / `best` / `diff`, and the MCP server.
7. **Before reporting to the scientist, a checklist:**
   - no failed or skipped cells, or they are reported
   - the seed count suits the test
   - corrected p-values and effect sizes, not raw means
   - a clean provenance
   - the sweep root path, so the run can be found again
8. **Going further.** Links to the specific guides (LLM eval, batteries,
   Study, launchers, multi-node, statistics), and `help(obj)` for exact
   signatures.

Each claim in sections 4 to 6 is verified when the guide is written, and the
code is exercised by the D7 test.

## Verification

- `tests/test_agent_guide.py` passes locally, with `--hypothesis-profile fast`.
  Breaking an API the guide uses (for example renaming a `run()` keyword in a
  scratch copy) makes it fail.
- `mkdocs build --strict` passes. The built `agents/index.html` contains the
  guide's code, and `llms.txt` and `llms-full.txt` exist at the site root,
  with the agents page listed first.
- `uv build --wheel` and an install into a clean venv give
  `python -m mushin guide --path` an existing file.
- `help(mushin)` shows the module docstring.
- ruff (format and check, including the guide's Markdown code blocks),
  codespell, the docs include guard, and the full test suite.

## Risks

- **The guide drifts from the API.** Mitigated by executing its code (D7) and
  keeping prose claims few and checkable.
- **A docs plugin breaks the strict build.** The plan checks it first, and D5
  has a fallback.
- **Absolute links point at the latest docs, not the installed version.** This
  is acceptable: the package carries the version-matched guide itself, and the
  links lead to deeper reading.
- **Import cost of `python -m mushin`.** It imports the package (about 1 s),
  which is acceptable for an occasional command.

## Follow-ups (not in this change)

- API reference pages for the exports that have none.
- Possible behavior changes surfaced while writing the guide (`best()`
  averaging over seeds, `show()` printing), each its own decision.
- Agent-specific packaging (a skill, MCP prompts) on top of the guide.
- Docstring entries for `run()`'s `working_dir` and `launcher`.
