# Design: drop Intel macOS (x86_64) support

**Date:** 2026-10-14
**Status:** Revised after spec review (13 findings) and plan review (8), all
accepted. Plan: [`2026-10-14-drop-intel-mac.md`](../plans/2026-10-14-drop-intel-mac.md).
**Related:** [drop Python 3.10, prepare 3.15](2026-10-14-python-311-315-design.md)
(#236). That change left Intel macOS with exactly one supported Python, 3.11.

## Problem

mushin keeps a second dependency branch just for Intel Macs. PyTorch has
published no x86_64 macOS wheels since torch 2.2, so on `darwin` + `x86_64`
mushin pins torch `>=2.2.0,<2.3`, numpy `>=1.24,<2` and torchvision
`>=0.17,<0.18`. Everywhere else it requires torch >= 2.4, numpy >= 2 and
torchvision >= 0.19. The branch costs:

- **3.15 blocker 4.** The numpy<2 fork makes the universal lock settle on
  scipy 1.17.1 for *every* fork, and that release has no cp315 wheel.
- **Dependabot cannot see torch, numpy or torchvision.** It cannot rewrite a
  marker-split declaration ("Declaration not found for torch!"), so
  `dependabot.yml` ignores all three. Its comment says to "Revisit if the
  marker split goes away (dropping Intel-Mac support would do it)".
- **The slowest CI leg.** `macos-15-intel` takes about 13 minutes.
- **A shrinking future.**
  - Since #236, Intel Macs support only Python 3.11.
  - GitHub's `macos-15-intel` is its last x86_64 image, and x86_64 macOS
    support ends when that image retires in fall 2027 (GitHub changelog,
    2025-09-19).
  - CPython moved x86_64-apple-darwin to tier 2 in August 2025 (PEP 11).

Apple Silicon is already fully supported and tested. CI runs the full suite on
`macos-latest` (arm64) for 3.13 and 3.14 on every PR, and `docs/install.md`
documents MPS training.

## Facts established (2026-10-14)

### The lock

"D1 edits" means `torch >= 2.4.0`, `numpy >= 2`, and `torchvision >= 0.19` in
both `detection` and `image`, with no markers. "Intel marker" means the exact
string `platform_machine == 'x86_64' and sys_platform == 'darwin'`.

- **D1 edits plus a plain `uv lock`:**
  - It removes torch 2.2.2, torchvision 0.17.2 and numpy 1.26.4, and adds
    nothing.
  - uv keeps the old forks as preferences, so **9 Intel markers remain**:
    3 top-level, where the fork count actually rises from 14 to 15, and 6
    package-level.
  - **`uv sync` on an x86_64 macOS interpreter** then fails up front, before
    installing anything, with "Distribution `torch==2.13.0` can't be installed
    because it doesn't have a source distribution or wheel for the current
    platform". uv's hint suggests adding Intel to `tool.uv.required-environments`.
- **D1 edits plus `[tool.uv] environments = ["sys_platform != 'darwin' or platform_machine != 'x86_64'"]`,
  then a plain `uv lock`:**
  - It removes **exactly five** versions that only the Intel fork used: numpy
    1.26.4, pandas 2.3.3, pytz 2026.2, torch 2.2.2 and torchvision 0.17.2.
    Nothing is added or upgraded.
  - Top-level forks go from 14 to 12, with **zero** Intel markers.
  - The lock records
    `supported-markers = ["platform_machine != 'x86_64' or sys_platform != 'darwin'"]`.
  - `uv sync` on an x86_64 macOS 3.11 fails immediately with
    `error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine != 'x86_64' or sys_platform != 'darwin'``.
- **An alternative that also cleans the lock, found in review.** Run the D1
  edits, delete the lock's top-level `resolution-markers` block, then run a
  plain `uv lock`. It removes the same five versions, upgrades nothing, and
  leaves zero Intel markers.
- **What remains after either clean lock:**
  - numpy keeps two versions: 2.4.6 for 3.11 and 2.5.0 for >= 3.12, because
    numpy 2.5 needs Python >= 3.12.
  - pandas drops to one version, 3.0.4. That release is yanked, and fixing it
    is not in scope (see below).
- **3.15 blocker 4 becomes a targeted upgrade.** Without the Intel fork,
  `uv lock --upgrade-package scipy --upgrade-package contourpy` gives scipy
  1.17.1 for 3.11 and **1.18.1, which has cp315 wheels, for >= 3.12**.
  contourpy splits the same way, 1.3.3 and 1.4.0. No marker-split requirement
  is needed.

### Users

- **The Intel fallback works on Python 3.10–3.12, but not 3.13+.**
  - mushin 0.14.2 pins torch 2.2.x on Intel, and torch 2.2.2's x86_64 macOS
    wheels are cp38–cp312 only.
  - On an x86_64 CPython 3.11 (under Rosetta),
    `uv pip install "mushin-py<0.15"` installs mushin 0.14.2 with torch 2.2.2
    and numpy 1.26.4.
  - `uv pip compile` with `--python-platform x86_64-apple-darwin` resolves on
    3.12 too, and fails on 3.13 and 3.14 ("only found wheels for torch (v2.2.2)
    with ... cp38 ... cp312").
  - The 0.14.x docs supported Intel on **3.10–3.11**, so that is the range to
    advertise.
- **The next release is at least 0.15.0.** The README says "patch releases
  never break", and #236 (dropping 3.10) already breaks users. So 0.14.x is the
  last series that supports Intel Macs.

### Dependabot, from dependabot-core source and this repo's history

- **When it raises floors.** For an unchanged `[project] description`,
  Dependabot treats the project as a library: `library?` is true when PyPI's
  `info.summary` equals the local description. It then widens ranges instead
  of raising floors, so runtime floors stay put. hydra-core, omegaconf and
  xarray got lock-only bumps.
  - When the description drifted from PyPI (#152, before 0.11.1 shipped), its
    next run (#154) raised every floor in its group.
  - It has also raised the `[build-system]` setuptools floor (#2, #155).
- **Which packages it can bump.** Its uv updater runs
  `uv lock --upgrade-package <name>==<version>` and resolves candidates at the
  lowest supported Python, 3.11. It has **never** bumped a package locked at
  more than one version: pandas, plus xarray, matplotlib and scikit-learn until
  #236 collapsed them. Open PR #238 now bumps those three.
- **After this change:**
  - **torch and torchvision** have one locked version each and are bumpable.
    `uv lock --upgrade-package torch==2.14.1` works lock-only and also moves
    torchvision to 0.29.1.
  - **numpy** (two locked versions) and **scipy** (1.18 needs Python >= 3.12)
    will **not** be bumped by Dependabot. They still need manual targeted
    upgrades.

### PESQ, re-tested because the old claim was wrong

- **The current code and docs claim more than is true.** They say pesq 0.0.4
  "fails to import under NumPy 2". pesq has four releases (0.0.1–0.0.4,
  2019–2022), all sdist-only.
- **With a C compiler, it works under NumPy 2.**
  - On `python:3.11-bookworm`, which has gcc, `pip install "numpy>=2" pesq==0.0.4`
    builds against numpy 2.4.6, and `pesq.pesq(16000, ref, deg, "wb")` returns
    4.554.
  - Review got the same result on macOS arm64 3.11, 3.13 and 3.14, and on
    x86_64 3.11.
- **Without a compiler, it does not install.** On `python:3.11-slim`, which has
  no gcc, the install fails.

### Everything that names Intel macOS, its pins, or two Mac architectures

`git grep`, excluding `CHANGELOG.md`, `uv.lock`, `docs/superpowers` and
notebooks, finds the following.

`pyproject.toml`:
- the 3.14 classifier comment (lines 25-27);
- the torch/numpy split and its comment (63-73);
- the "non-Intel" floor notes for pandas (82-84), matplotlib (89-92), scipy
  (99-101) and scikit-learn (176-178);
- the `detection` torchvision split and its comment (106-114);
- the `image` torchvision split and its "platform-split torch floors" comment
  (120-127). That comment contains no "Intel", which an Intel-only grep would
  miss;
- the PESQ note (132-134).

Other files:
- **`.github/workflows/ci.yml`:** the `macos-15-intel` leg, the macOS comment,
  and the 3.14 comment.
- **`.github/workflows/python-315-preview.yml`:** the matrix comment.
- **`.github/dependabot.yml`:** the torch/numpy/torchvision ignore and its
  comment.
- **`.github/PULL_REQUEST_TEMPLATE.md`:** "not only Intel-macOS".
- **`CONTRIBUTING.md`:** "Contributors on Intel macOS cannot install
  torch > 2.2".
- **`docs/install.md`:**
  - the support-matrix rows;
  - the Intel macOS note;
  - "On Intel macOS that leaves 3.11 as the only supported version".
- **`changes/236.removed.md`** (unreleased): the same Intel sentence.
- **Other false PESQ statements:** `docs/guides/custom.md:196` ("its only
  release is NumPy-2-incompatible"). `.github/workflows/ci.yml:110` also lists
  pesq among the extras' C-extension deps, but pesq is in no extra.
- **PESQ advice that names a "NumPy-1.x environment":**
  `src/mushin/benchmark/_metrics.py:236-239` and
  `docs/guides/batteries.md:582-583`. Once NumPy 2 is required everywhere, such
  an environment cannot run mushin, and the premise behind the advice is false
  anyway (see PESQ above).
- **The 3.15 spec's D4 checklist.** Live comments in `pyproject.toml` and the
  preview workflow point to it. It says the promotion PR bumps torch,
  torchvision and numpy "by hand, since Dependabot ignores them". It also
  points to "the torch and numpy note in `dependabot.yml`", which this change
  deletes.

## Decisions

### D1. One dependency branch

- **Declarations:** `torch >= 2.4.0`, `numpy >= 2`, and
  `torchvision >= 0.19` in `detection` and `image`, all without markers.
- **The floors themselves do not move.** They were already the non-Intel
  floors.
  - `min-versions` validates the numpy floor. It resolves torch **2.4.1**, not
    2.4.0: the lock resolves every extra together, and torchvision 0.19.0 caps
    numpy<2. So the torch 2.4.0 floor itself is not tested, and no comment may
    claim it is.
  - **The torchvision floor is not floor-tested.** `min-versions` installs
    only `--extra eval`, so no comment may claim that job verifies it.
- **Every inventoried `pyproject.toml` comment is rewritten:**
  - The torch/numpy block gives the NumPy-2 reason, plus one sentence that
    Intel macOS is unsupported because PyTorch ships no x86_64 macOS wheels
    after 2.2.
  - The 3.14 classifier comment loses its Intel sentence.
  - The pandas, matplotlib, scipy and scikit-learn notes say "the NumPy-2
    floor" instead of "the non-Intel floor", and lose their "still supports
    NumPy 1.x for Intel-mac" clauses.
  - Both torchvision comments say the 0.19 floor pairs with torch 2.4.
  - The PESQ note gives the real reason PESQ is not bundled (D5).

### D2. Scope the lock with `[tool.uv] environments`

Add
`environments = ["sys_platform != 'darwin' or platform_machine != 'x86_64'"]`
under `[tool.uv]`, with a comment.

**Placement:** before `[tool.pytest.ini_options]`. It must not go above
`[tool.setuptools.packages.find]`, because `ci.yml` points to "the note above
[tool.setuptools.packages.find]", meaning the pycocotools note.

**Why:**
- **Clearer failure.** It turns uv's up-front "no wheel for torch" error,
  whose hint invites adding Intel back through `required-environments`, into
  an explicit statement that the platform is not supported.
- **It stays clean.** The support boundary is recorded in the tooling, so a
  future lock cannot quietly grow Intel forks again.
- **It cleans the lock now.** Exactly five Intel-only versions drop, with
  nothing upgraded.
- **Nothing outside uv sees it.** pip, `uv build` and the published metadata
  ignore `[tool.uv]`.

**Rejected:** deleting `resolution-markers` and re-locking. That also cleans
the lock, but it records no boundary, so Intel users get the less clear error.
And hand-editing the lock is a step nobody could reproduce from the repo.

### D3. CI: replace the Intel leg with Apple Silicon on the oldest Python

Replace `{ os: macos-15-intel, python-version: "3.11" }` with
`{ os: macos-latest, python-version: "3.11" }`.

- macOS then covers the oldest and the newest supported Python, all on arm64.
- The suite passes locally on arm64 3.11 (763 passed).
- The macOS comment and the 3.14 comment are rewritten.

### D4. Dependabot: stop ignoring torch, numpy and torchvision

Delete the three entries and their comment, as that comment asks. Keep the
`setuptools` entry.

The new, short comment states what is true after this change:
- torch and torchvision get normal grouped updates;
- numpy (locked at two versions, 3.11 / 3.12+) and scipy still need manual
  `uv lock --upgrade-package <name>`. Dependabot tracks only a package's
  *lowest* locked version and resolves at Python 3.11. If it ever proposes a
  bump for such a package, its `uv lock --upgrade-package name==X` collapses
  every entry to X, so the lock diff needs checking;
- Dependabot keeps runtime floors only while `[project] description` matches
  the PyPI summary. When the two drift, it raises floors (#154), so check the
  `pyproject.toml` diff on Dependabot PRs after changing the description.

Un-ignoring numpy is still right. It produces no failing runs, and it no longer
blocks security updates for numpy.

### D5. Docs and messages

- **`docs/install.md`:**
  - **The matrix collapses to one row:** "Linux / Windows / macOS (Apple
    Silicon) | 3.11 – 3.14 | ≥ 2.4 | ≥ 2".
  - **The Intel macOS note:** Intel is unsupported because PyTorch ships no
    x86_64 macOS wheels after 2.2. mushin 0.14.x is the last series that
    supports it, **tested there on Python 3.10 and 3.11**:
    `python3.11 -m pip install "mushin-py<0.15"`. torch 2.2 has no wheels for
    Python 3.13+.
  - **The 3.10 note** loses its Intel sentence.
- **`changes/236.removed.md`** (unreleased, ships in the same release) loses
  the Intel sentence, so the release notes do not contradict themselves.
  `changes/` edits pass the changelog CI. Only direct `CHANGELOG.md` edits are
  rejected.
- **`CONTRIBUTING.md`:** drop the Intel-macOS contributor sentence.
- **PR template:** "not only Intel-macOS" becomes "not only your local
  environment".
- **The `ci.yml` battery comment:** drop pesq from its C-extension list.
- **PESQ** (`_metrics.py`, `batteries.md`, `custom.md` and the
  `pyproject.toml` audio note): replace the false "fails under NumPy 2 / use a
  NumPy-1.x environment" claims with the tested facts:
  - every `pesq` release (latest 0.0.4, 2022) ships only as source and needs a
    C compiler to install, so it is not bundled in `audio`;
  - with a compiler, it builds and runs under NumPy 2;
  - users can `pip install pesq` and add it through a custom Task.
- **`python-315-preview.yml`:** drop the Intel sentence.
- **Changelog:** a `removed` fragment giving the reason, the last supported
  series and the Python range, and the pin.

### D6. Update the 3.15 promotion checklist

The 3.15 spec's D4 gets a dated note:
1. Blocker 4 now resolves with a targeted
   `uv lock --upgrade-package scipy --upgrade-package contourpy` (verified),
   as an **explicit promotion step**, because Dependabot cannot make this split.
2. Dependabot now bumps torch and torchvision. The promotion PR confirms those
   versions have cp315 wheels, or bumps them by hand. numpy is still bumped by
   hand.
3. The `dependabot.yml` marker-split note that the checklist cites no longer
   exists.

This PR performs **none** of those upgrades, under the no-incidental-upgrades
rule.

## Non-goals

- **`_provenance.py`'s Intel-CPU MPS branch.** It stops provenance from
  calling an Intel Mac with a Metal GPU "Apple Silicon". That is correct for
  any machine mushin runs on, and it is not a support claim.
- **The `mcp/server.py` NumPy 1.x unpickling path.** It reads datasets that
  older installs saved.
- **Upgrading scipy, contourpy, numpy, pandas or anything else.** The locked
  pandas 3.0.4 is yanked. After this change it is single-version, so Dependabot
  should now be able to bump it. That gets checked on the next Dependabot run,
  not here.
- **Changing Dependabot's `versioning-strategy`.** It is a repo-wide policy
  change. D4's comment documents the drift risk instead.
- **The pre-existing `setuptools` ignore comment.** It says the updater "does
  not edit `[build-system]` at all", which #2 and #155 contradict. That is
  unrelated to Intel and is left for a separate fix.
- Committed notebook outputs, and historical documents under
  `docs/superpowers/`.

## Verification

- **Lock diff by package against `origin/main`:** exactly the five removals
  and nothing added. Zero Intel markers (the exact string above), and the
  `supported-markers` line present.
- **`uv sync` on the Rosetta x86_64 3.11** in a scratch copy fails with the
  "not compatible with the lockfile's supported environments" error.
- **The local gate:** ruff, format, codespell, `mkdocs build --strict`, and
  the full suite on arm64 3.11 with every battery extra.
- **`make test-lowest`:** the floors are unchanged, so expect the same result
  as #236's run (752 passed).
- **Leftover-wording grep:**
  `git grep -nIiE "intel|x86_64|platform-split|split by platform|marker.split|non-Intel|numpy-1|numpy 1\.x"`.
  Only the non-goal sites, the new "unsupported" statements and the Makefile's
  "x86_64 Linux" may remain.
- **PR CI:** the `macos-latest` 3.11 leg passes, and `ci-ok` is green.
