# Design: drop Intel macOS (x86_64) support

**Date:** 2026-10-14
**Status:** Draft. Pending adversarial review, then an implementation plan.
**Related:** [drop Python 3.10, prepare 3.15](2026-10-14-python-311-315-design.md)
(#236). That change left Intel macOS with exactly one supported Python, 3.11.

## Problem

mushin keeps a second dependency branch just for Intel Macs. PyTorch has
published no x86_64 macOS wheels since torch 2.2, so on `darwin` + `x86_64`
mushin pins torch `>=2.2.0,<2.3`, numpy `>=1.24,<2` and torchvision
`>=0.17,<0.18`. Everywhere else it requires torch >= 2.4, numpy >= 2 and
torchvision >= 0.19. That branch costs real things:

- **It blocks 3.15 (blocker 4 in the 3.15 spec).** The numpy<2 fork makes the
  universal lock settle on scipy 1.17.1 for *every* fork, and scipy 1.17.1 has
  no cp315 wheel.
- **It blinds Dependabot to torch, numpy and torchvision.** Dependabot cannot
  rewrite a marker-split declaration ("Declaration not found for torch!"), so
  `dependabot.yml` ignores all three. Its comment says to "Revisit if the
  marker split goes away (dropping Intel-Mac support would do it)".
- **It costs the slowest CI leg.** `macos-15-intel` takes about 13 minutes.
- **It has a narrowing future.** Since #236, Intel Macs support only Python
  3.11. GitHub's `macos-15-intel` is its last x86_64 image, due to retire in
  fall 2027. CPython is discussing moving Intel Macs to tier 2.

Apple Silicon is already fully supported and tested. CI runs the full suite on
`macos-latest` (arm64) for 3.13 and 3.14 on every PR, and `docs/install.md`
documents MPS training.

## Facts established (2026-10-14)

### The lock

The scratch experiment replaced the three marker-split pairs with single
declarations: `torch >= 2.4.0`, `numpy >= 2`, and `torchvision >= 0.19` in both
the `detection` and `image` extras.

- **Plain `uv lock` alone:** it removes exactly torch 2.2.2, torchvision
  0.17.2 and numpy 1.26.4, and adds nothing. But uv keeps the old forks as
  preferences, so 20 `darwin`/`x86_64` resolution markers remain in the lock.
- **Plus `[tool.uv] environments`:** with
  `environments = ["sys_platform != 'darwin' or platform_machine != 'x86_64'"]`
  added, a plain `uv lock` removes **exactly five** versions that only the
  Intel fork used: numpy 1.26.4, pandas 2.3.3, pytz 2026.2, torch 2.2.2 and
  torchvision 0.17.2.
  - Nothing is added or upgraded.
  - Resolution markers go from 14 to 12, with zero `darwin`/`x86_64` markers
    left.
  - The lock records
    `supported-markers = ["platform_machine != 'x86_64' or sys_platform != 'darwin'"]`.
- **`uv sync` then refuses Intel Macs outright.** In the scoped scratch lock,
  `uv sync` on an x86_64 macOS 3.11 fails immediately with
  `error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine != 'x86_64' or sys_platform != 'darwin'``.
- **3.15 blocker 4 goes away.** Without the Intel fork,
  `uv lock --upgrade-package scipy --upgrade-package contourpy` gives scipy
  1.17.1 for 3.11 and **1.18.1 (which has cp315 wheels) for >= 3.12**, and
  contourpy 1.3.3 plus 1.4.0 the same way. No marker-split requirement is
  needed.

### Users

- **Intel-Mac users keep a working path.** On an x86_64 CPython 3.11 (run
  under Rosetta), `uv pip install "mushin-py<0.15"` installs mushin 0.14.2
  with torch 2.2.2 and numpy 1.26.4.
- **Why "<0.15" is right.** The README's policy is that "patch releases never
  break", and #236 (dropping 3.10) already breaks users. So the next release is
  at least 0.15.0, and 0.14.x is the last series that supports Intel Macs.

### Dependabot

- In this repo's history, Dependabot has raised floors only in the dev
  dependency groups: ruff, pre-commit, hypothesis, jupytext, mkdocstrings.
- Runtime dependencies such as pandas, xarray and hydra-core get lock-only
  bumps.
- So un-ignoring torch, numpy and torchvision should produce lock bumps, not
  floor raises.

### Everything that names Intel macOS or its pins

`git grep`, excluding `CHANGELOG.md`, `uv.lock`, `docs/superpowers` and
notebooks, finds:
- `pyproject.toml`:
  - the torch/numpy split and its comment;
  - the torchvision split in `detection` and `image`;
  - "non-Intel" floor notes for pandas, matplotlib, scipy, scikit-learn and
    the PESQ note;
  - the 3.14 classifier comment.
- `.github/workflows/ci.yml`: the `macos-15-intel` leg and two comments.
- `.github/workflows/python-315-preview.yml`: one comment.
- `.github/dependabot.yml`: the torch/numpy/torchvision ignore and its long
  comment.
- `.github/PULL_REQUEST_TEMPLATE.md`: "not only Intel-macOS".
- `CONTRIBUTING.md`: "Contributors on Intel macOS cannot install torch > 2.2".
- `docs/install.md`:
  - the support-matrix rows;
  - the Intel macOS note;
  - "On Intel macOS that leaves 3.11 as the only supported version".
- `changes/236.removed.md` (unreleased): the same Intel sentence.
- **PESQ advice that becomes impossible.** Two places tell users to add PESQ
  "on a NumPy-1.x environment": `src/mushin/benchmark/_metrics.py:237-239` and
  `docs/guides/batteries.md:583`. Once NumPy 2 is required everywhere, mushin
  cannot run in such an environment.

## Decisions

### D1. One dependency branch

- `pyproject.toml` declares `torch >= 2.4.0`, `numpy >= 2`, and
  `torchvision >= 0.19` in both `detection` and `image`, all without markers.
- The comments are rewritten to give the NumPy-2 reason, plus one sentence
  saying Intel macOS is unsupported because PyTorch ships no x86_64 macOS
  wheels after 2.2.
- **The floors themselves do not move.** They were already the non-Intel
  floors, and `min-versions` (Linux) already validates them.

### D2. Scope the lock with `[tool.uv] environments`

Add
`[tool.uv] environments = ["sys_platform != 'darwin' or platform_machine != 'x86_64'"]`,
with a comment.

**Why:**
- It is the only way, short of a fresh lock (which upgrades everything), to
  drop the dead Intel forks from `uv.lock`.
- It turns `uv sync` on an Intel Mac into an immediate, explicit
  "unsupported environment" error, instead of a confusing "no wheel for torch"
  failure halfway through the install.
- It is uv-only configuration. pip and the published metadata never see it.

**Rejected:**
- *Leaving the lock's stale Intel forks.* That is harmless at runtime, but it
  is misleading cruft, and the next contributor would wonder why it is there.
- *A fresh lock.* That is a mass upgrade, which contradicts this repo's
  lock-hygiene rule.

### D3. CI: replace the Intel leg with Apple Silicon on the oldest Python

Replace `{ os: macos-15-intel, python-version: "3.11" }` with
`{ os: macos-latest, python-version: "3.11" }`, and rewrite the macOS comment.

- macOS then keeps a leg on the oldest *and* the newest supported Python, all
  on Apple Silicon.
- The suite already passes on arm64 3.11 locally (763 passed).
- The 3.14 comment's "Intel-mac is deliberately absent" sentence is removed.

### D4. Dependabot: stop ignoring torch, numpy and torchvision

Delete those three ignore entries and their comment, as that comment itself
asks. Keep the `setuptools` ignore, whose reason is unrelated.

- **Expected effect:** lock-only weekly bumps. The locked torch is 2.13.0;
  2.14.1 is out.
- **Risk:** Dependabot raises a runtime floor in `pyproject.toml`. History says
  it does not, but the first torch bump PR should be checked for that before
  merging. The PR body will say so.

### D5. Docs and messages

- `docs/install.md`:
  - The matrix collapses to one row: "Linux / Windows / macOS (Apple Silicon)
    | 3.11 – 3.14 | ≥ 2.4 | ≥ 2".
  - The Intel macOS note becomes: unsupported, because of PyTorch's x86_64
    macOS wheels; 0.14.x is the last series that supports it;
    `pip install "mushin-py<0.15"`.
  - The 3.10 note loses its Intel sentence.
- `changes/236.removed.md`, which is unreleased and ships in the same release,
  loses the Intel sentence. Otherwise the release notes would contradict
  themselves.
- `CONTRIBUTING.md`: drop the Intel-macOS contributor sentence.
- PR template: drop "not only Intel-macOS".
- PESQ advice, in `_metrics.py` and `batteries.md`: drop the "NumPy-1.x
  environment" suggestion. Say instead that PESQ is omitted because the only
  released `pesq` fails under NumPy 2, which mushin requires.
- `python-315-preview.yml`: drop the Intel sentence.
- Changelog: a `removed` fragment giving the reason, the last supported series
  and the pin.

### D6. Update the 3.15 promotion checklist

The 3.15 spec's D4 is referenced from live comments in `pyproject.toml` and
the preview workflow. It gets a dated note:
- blocker 4 now resolves with a targeted
  `uv lock --upgrade-package scipy --upgrade-package contourpy`, verified as
  above;
- no marker split is needed.

This PR does **not** perform that upgrade. The no-incidental-upgrades rule
applies, and Dependabot will now bump scipy on its own.

## Non-goals

- **`_provenance.py`'s Intel-CPU MPS branch.** It stops provenance from
  calling an Intel Mac with a Metal GPU "Apple Silicon". That is correct for
  any machine mushin happens to run on, conda-forge torch builds for example,
  and it is not part of the support claim.
- **The `mcp/server.py` NumPy 1.x unpickling path.** It reads datasets that
  older installs saved, which is about data compatibility, not platform
  support.
- **Upgrading scipy, contourpy or anything else** (see D6).
- Committed notebook outputs, and historical documents under
  `docs/superpowers/`.

## Verification

- **Lock diff by package against `origin/main`:** exactly the five removals
  above and nothing added. Zero `platform_machine == 'x86_64' and
  sys_platform == 'darwin'` markers, and the `supported-markers` line present.
- **`uv sync` refuses an x86_64 macOS interpreter.** Use the Rosetta 3.11 in
  a scratch copy, and expect a "not compatible with the lockfile's supported
  environments" error.
- **The local gate:** ruff, format, codespell, `mkdocs build --strict`, and
  the full suite on arm64 3.11 with every battery extra.
- **`make test-lowest`:** the floors are unchanged, so its result should match
  #236's run (752 passed).
- **`git grep` for Intel/x86_64 mentions:** only the non-goal sites and the
  new "unsupported" statements remain.
- **PR CI:** the `macos-latest` 3.11 leg passes, and `ci-ok` is green.
