# Drop Intel macOS Support: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Intel-macOS (darwin + x86_64) dependency branch so mushin
has one dependency set: torch >= 2.4, NumPy 2, torchvision >= 0.19. Scope the
lock to the supported platforms, move the macOS CI leg to Apple Silicon, and
make every user-facing statement true.

**Architecture:** This is metadata, CI and docs only. The single declarations
replace marker-split pairs. `[tool.uv] environments` scopes `uv.lock`, which is
regenerated with a plain `uv lock`. Read
[the spec](../specs/2026-10-14-drop-intel-mac-design.md) first. It holds every
fact and decision this plan relies on.

**Tech Stack:** uv, GitHub Actions, Dependabot, towncrier, ruff, pytest, Docker.

**Conventions:**
- No Claude attribution anywhere.
- Never run a bare `uv lock --upgrade`. Use a plain `uv lock`, then diff by
  package against `origin/main`.
- Fragments start as `changes/+slug.<type>.md` and are renamed to
  `<PR>.<type>.md` once the PR exists.
- **Patch with unique anchors.** In each step, replace the quoted text
  exactly. Use a short Python script that asserts each anchor occurs once,
  rather than hand edits.

---

### Task 1: Collapse the dependency branch and scope the lock

**Files:** `pyproject.toml`, `uv.lock` (regenerated)

- [ ] **Step 1: The 3.14 classifier comment.** Replace

  ```toml
      # 3.14 is classified now that its CI legs (linux/windows/macos-arm64) are
      # required to pass rather than continue-on-error. Intel-mac is excluded from
      # that matrix: it caps at torch 2.2.2, which has no 3.14 wheels.
  ```
  with
  ```toml
      # 3.14 is classified now that its CI legs (linux/windows/macos-arm64) are
      # required to pass rather than continue-on-error.
  ```

- [ ] **Step 2: torch and numpy.** Replace the block from
  `    # torch floors are split by platform.` through
  `    "numpy >= 2 ; sys_platform != 'darwin' or platform_machine != 'x86_64'",`
  (ten lines) with

  ```toml
      # torch floor is 2.4: mushin requires NumPy 2 (below), which is
      # ABI-incompatible with torch built against NumPy 1.x, and PyTorch first
      # ships NumPy-2-compatible wheels in 2.4.0 ("Numpy is not available" below
      # that). Intel macOS (x86_64) is not supported: PyTorch publishes no x86_64
      # macOS wheels after 2.2 (see [tool.uv] environments). Both floors are
      # verified by the min-version CI job.
      "torch >= 2.4.0",
      "numpy >= 2",
  ```

- [ ] **Step 3: The pandas note.** Replace

  ```toml
      # pandas 2.2.2 is the first release officially compatible with NumPy 2 (the
      # non-Intel floor); older pandas imports under NumPy 2 only by luck (ABI
      # "size changed" warning). Verified by the min-version CI job.
  ```
  with
  ```toml
      # pandas 2.2.2 is the first release officially compatible with NumPy 2;
      # older pandas imports under NumPy 2 only by luck (ABI "size changed"
      # warning). Verified by the min-version CI job.
  ```

- [ ] **Step 4: The matplotlib note.** Replace

  ```toml
  # matplotlib >= 3.9 works with NumPy 2 (the non-Intel floor) and still supports
  # NumPy 1.x for the Intel-mac branch. Releases up to 3.8.3 do not work with
  # NumPy 2: 3.7.3-3.8.3 cap numpy<2, and older wheels fail its ABI ("numpy.core.
  # multiarray failed to import"). Verified by the min-version CI job.
  ```
  with
  ```toml
  # matplotlib >= 3.9 works with NumPy 2. Releases up to 3.8.3 do not: 3.7.3-3.8.3
  # cap numpy<2, and older wheels fail its ABI ("numpy.core.multiarray failed to
  # import"). Verified by the min-version CI job (matplotlib is in the dev group).
  ```

- [ ] **Step 5: The scipy note.** Replace

  ```toml
  # incompatible, `_ARRAY_API not found`); it still supports NumPy 1.x for Intel-mac.
  # Verified by the min-version CI job.
  ```
  with
  ```toml
  # incompatible, `_ARRAY_API not found`). Verified by the min-version CI job.
  ```

- [ ] **Step 6: The detection comment and dependency.** Replace

  ```toml
  # pycocotools; gated so the core install stays lean. torchvision floors pair with
  # the platform-split torch floors above (0.17 ↔ torch 2.2 on Intel-mac, 0.19 ↔
  # torch 2.4 elsewhere). pycocotools floor is 2.0.7 — the first release with wheels
  # (2.0.0 has no Cython build-dep and fails to build under the min-versions job).
  ```
  with
  ```toml
  # pycocotools; gated so the core install stays lean. The torchvision floor (0.19)
  # pairs with the torch >= 2.4 floor above; no CI job floor-tests it, because
  # min-versions installs only the eval extra. pycocotools floor is 2.0.7 — the
  # first release with wheels (2.0.0 has no Cython build-dep and fails to build).
  ```

- [ ] **Step 7: The image comment.** Replace

  ```toml
  # this extra as a whole. torchvision floors pair with the platform-split torch
  # floors above.
  ```
  with
  ```toml
  # this extra as a whole. The torchvision floor pairs with the torch floor above.
  ```

- [ ] **Step 8: Both torchvision splits.** Each occurs twice, in `detection`
  and in `image`. Delete both copies of

  ```toml
      "torchvision >= 0.17,<0.18 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
  ```
  and replace both copies of
  `    "torchvision >= 0.19 ; sys_platform != 'darwin' or platform_machine != 'x86_64'",`
  with `    "torchvision >= 0.19",`. Assert each anchor occurs exactly twice.

- [ ] **Step 9: The PESQ note.** Replace

  ```toml
  # raises a clear error if used without it). PESQ is intentionally excluded: the
  # only released `pesq` (0.0.4) fails to import under NumPy 2 and has no fixed
  # release, so it would crash on the supported non-Intel platforms.
  ```
  with
  ```toml
  # raises a clear error if used without it). PESQ is intentionally excluded: the
  # only `pesq` release (0.0.4, 2022) is source-only and needs a C compiler to
  # install. With one it builds and runs under NumPy 2; add it via a custom Task.
  ```

- [ ] **Step 10: The scikit-learn note.** Replace

  ```toml
      # Floor is 1.4.2 — the first release compatible with NumPy 2 (the non-Intel
      # floor); earlier scikit-learn is ABI-incompatible with NumPy 2. The
  ```
  with
  ```toml
      # Floor is 1.4.2 — the first release compatible with NumPy 2; earlier
      # scikit-learn is ABI-incompatible with NumPy 2. The
  ```

- [ ] **Step 11: Add `[tool.uv]`.** Insert this immediately before the line
  `[tool.setuptools.packages.find]`, keeping a blank line before it:

  ```toml
  [tool.uv]
  # The platforms uv.lock resolves for. Intel macOS (darwin + x86_64) is
  # excluded: PyTorch publishes no x86_64 macOS wheels after 2.2, and mushin
  # requires torch >= 2.4. Without this, `uv sync` there fails on a missing torch
  # wheel; with it, uv states that the platform is unsupported. pip and the
  # published package metadata ignore this table.
  environments = ["sys_platform != 'darwin' or platform_machine != 'x86_64'"]

  ```

- [ ] **Step 12: Confirm no marker split remains.**

  Run: `grep -nE "platform_machine|x86_64|Intel|non-Intel|platform-split" pyproject.toml`
  Expected: only the Step 2 comment ("Intel macOS (x86_64) is not
  supported ...") and the Step 11 `[tool.uv]` comment and `environments` line.

- [ ] **Step 13: Lock, then diff by package.**

  Run: `uv lock`
  Expected:
  - `Removed pytz v2026.2`;
  - `Updated` lines that only drop a version: numpy `v1.26.4, v2.4.6, v2.5.0 -> v2.4.6, v2.5.0`,
    pandas `v2.3.3, v3.0.4 -> v3.0.4`, torch `v2.2.2, v2.13.0 -> v2.13.0`,
    torchvision `v0.17.2, v0.28.0 -> v0.28.0`.

  **Stop condition:** any `Updated` line whose right-hand side has a version
  not on its left.

  Then run the by-package diff from the 3.10 plan (Task 1 Step 7), with
  `/tmp/uv.lock.main` from `git show origin/main:uv.lock`.
  Expected: exactly these 5 removals, each `added=[]`:
  - numpy 1.26.4
  - pandas 2.3.3
  - pytz 2026.2
  - torch 2.2.2
  - torchvision 0.17.2

  Then:
  ```bash
  grep -c "platform_machine == 'x86_64' and sys_platform == 'darwin'" uv.lock   # expect 0
  grep -n "supported-markers" uv.lock                                          # expect the one line
  ```

- [ ] **Step 14: Re-sync the dev environment.** It is already arm64 3.11.

  Run: `uv sync --extra eval --extra detection --extra image --extra audio`
  Expected: it removes torch 2.2/numpy 1.x entries only if present (the arm64
  env never had them), so effectively no change.

- [ ] **Step 15: Commit.**
  ```bash
  git add pyproject.toml uv.lock
  git commit -m "build: drop Intel macOS: one torch/numpy/torchvision branch, lock scoped to supported platforms"
  ```

### Task 2: Correct the PESQ advice

**Files:** `src/mushin/benchmark/_metrics.py:236-239`, `docs/guides/batteries.md:582-583`

- [ ] **Step 1: The docstring.** In `_metrics.py`, replace

  ```python
      PESQ is intentionally omitted: the only released ``pesq`` package (0.0.4) fails
      to import under NumPy 2 (the supported non-Intel platform) and has no fixed
      release, so it cannot be shipped reliably. Add it via a custom Task on a
      NumPy-1.x environment if needed."""
  ```
  with
  ```python
      PESQ is intentionally omitted: the only ``pesq`` release (0.0.4, 2022) is
      source-only and needs a C compiler to install, so it cannot be bundled
      reliably. With a compiler it builds and runs under NumPy 2; add it via a
      custom Task if needed."""
  ```

- [ ] **Step 2: The guide.** In `batteries.md`, replace

  ```markdown
  (PESQ is intentionally omitted; its only released package fails to import under
  NumPy 2; add it via a custom task on a NumPy-1.x environment if you need it.)
  ```
  with
  ```markdown
  (PESQ is intentionally omitted: the only `pesq` release is source-only and needs
  a C compiler to install. With one, `pip install pesq` works under NumPy 2, and
  you can add PESQ through a custom task.)
  ```

- [ ] **Step 3: Check nothing asserts the old text.**

  Run: `git grep -n "NumPy-1.x\|non-Intel" -- src tests docs/guides`
  Expected: no output.

- [ ] **Step 4: Commit.**
  ```bash
  git add src/mushin/benchmark/_metrics.py docs/guides/batteries.md
  git commit -m "docs: PESQ is omitted because it is source-only, not because of NumPy 2"
  ```

### Task 3: CI on Apple Silicon only

**Files:** `.github/workflows/ci.yml`, `.github/workflows/python-315-preview.yml`

- [ ] **Step 1: The `ci.yml` macOS legs.** Replace

  ```yaml
            # macOS: researchers' most common dev platform, and the platform-split
            # dependency branches in pyproject.toml (Intel-mac torch<2.3/numpy<2 vs
            # arm64 NumPy-2) are exercised nowhere else. One arm64 leg on the newest
            # supported Python, one x86_64 leg on the oldest (3.11).
            - { os: macos-latest, python-version: "3.13", experimental: false }
            - { os: macos-15-intel, python-version: "3.11", experimental: false }
            # 3.14 is a supported, advertised version: these legs block like any
            # other. Intel-mac is deliberately absent — it caps at torch 2.2.2,
            # which has no 3.14 wheels.
  ```
  with
  ```yaml
            # macOS (Apple Silicon): researchers' most common dev platform, and its
            # arm64 wheels are exercised nowhere else. Legs on the oldest (3.11)
            # and a recent (3.13) Python, plus 3.14 below. Intel macOS is not
            # supported (no x86_64 macOS torch wheels after 2.2).
            - { os: macos-latest, python-version: "3.11", experimental: false }
            - { os: macos-latest, python-version: "3.13", experimental: false }
            # 3.14 is a supported, advertised version: these legs block like any
            # other.
  ```

- [ ] **Step 2: The preview matrix comment.** Replace

  ```yaml
          # The platforms the 3.14 legs cover. Intel-mac is absent: it caps at
          # torch 2.2.x, which has no cp315 wheels.
  ```
  with
  ```yaml
          # The platforms the 3.14 legs cover.
  ```

- [ ] **Step 3: Validate the YAML and check what remains.**

  Run:
  ```bash
  uv run --no-project --with pyyaml python -I -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); inc=d['jobs']['test']['strategy']['matrix']['include']; print(sorted((i['os'], i['python-version']) for i in inc))"
  git grep -nIiE "intel|x86_64" -- .github/workflows/
  ```
  Expected:
  - The include list is
    `[('macos-latest','3.11'), ('macos-latest','3.13'), ('macos-latest','3.14'), ('ubuntu-latest','3.14'), ('windows-latest','3.14')]`.
  - The grep prints only the new ci.yml comment line "Intel macOS is not ...".

- [ ] **Step 4: Commit.**
  ```bash
  git add .github/workflows/ci.yml .github/workflows/python-315-preview.yml
  git commit -m "ci: replace the Intel-mac leg with Apple Silicon on 3.11"
  ```

### Task 4: Dependabot

**Files:** `.github/dependabot.yml`

- [ ] **Step 1: Replace the ignore block.** Replace everything from the line
  `    ignore:` through `      - dependency-name: "torchvision"` (the
  torch/numpy/torchvision comment plus three entries) and the blank line after
  it with:

  ```yaml
      # torch, torchvision and numpy used to be ignored here: Intel macOS needed
      # its own marker-split pins for them, which Dependabot cannot rewrite. That
      # split is gone, so they update normally, with two limits worth knowing:
      #   - Dependabot never bumps a package that uv.lock holds at more than one
      #     version, and it resolves at the lowest supported Python (3.11). So
      #     numpy (locked once for 3.11 and once for 3.12+) and scipy (1.18+
      #     needs Python 3.12) still need a manual
      #     `uv lock --upgrade-package <name>`.
      #   - Runtime floors in pyproject.toml stay put only while
      #     [project].description matches the PyPI summary of the latest
      #     release; when they differ, Dependabot raises floors (#154). Check the
      #     pyproject.toml diff on Dependabot PRs after editing the description.
      ignore:
  ```
  The `setuptools` comment and entry that followed stay exactly as they are,
  now as the only ignore.

- [ ] **Step 2: Validate.**

  Run: `uv run --no-project --with pyyaml python -I -c "import yaml; d=yaml.safe_load(open('.github/dependabot.yml')); print([i['dependency-name'] for i in d['updates'][0]['ignore']])"`
  Expected: `['setuptools']`.

- [ ] **Step 3: Commit.**
  ```bash
  git add .github/dependabot.yml
  git commit -m "chore(dependabot): stop ignoring torch, numpy and torchvision"
  ```

### Task 5: Docs, the contributor guide and the PR template

**Files:** `docs/install.md`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: The install.md matrix.** Replace the two rows

  ```markdown
  | Linux / Windows / non-Intel macOS | 3.11 – 3.14 | ≥ 2.4 | ≥ 2 |
  | Intel macOS (x86_64) | 3.11 | 2.2.x | 1.x |
  ```
  with
  ```markdown
  | Linux / Windows / macOS (Apple Silicon) | 3.11 – 3.14 | ≥ 2.4 | ≥ 2 |
  ```

- [ ] **Step 2: The install.md Intel note.** Replace

  ```markdown
  - **Intel macOS**: Apple has not shipped PyTorch wheels past 2.2.x for the
    x86_64 architecture. `mushin` supports this platform at torch 2.2.x and
    NumPy 1.x, but Python 3.12+ is not available there because NumPy 2 is
    required for Python 3.12 wheels and is ABI-incompatible with torch 2.2.x.
  ```
  with
  ```markdown
  - **Intel macOS (x86_64)** is not supported: PyTorch publishes no x86_64 macOS
    wheels after torch 2.2, and `mushin` requires torch ≥ 2.4 and NumPy 2. The
    last release series that supports Intel Macs is 0.14.x, on Python 3.10 or
    3.11 (torch 2.2 has no wheels for Python 3.13+):
    `python3.11 -m pip install "mushin-py<0.15"`.
  ```

- [ ] **Step 3: The install.md 3.10 note.** Replace

  ```markdown
    and `mushin` requires Python ≥ 3.11. On Intel macOS that leaves 3.11 as
    the only supported version.
  ```
  with
  ```markdown
    and `mushin` requires Python ≥ 3.11.
  ```

- [ ] **Step 4: CONTRIBUTING.** Replace

  ```markdown
  source of truth. Contributors on Intel macOS cannot install torch > 2.2; a Linux
  container or CI is the reliable way to test newer torch there.
  ```
  with `source of truth.`.

- [ ] **Step 5: The PR template.** Replace
  `      torch (CI matrix or a Linux container), not only Intel-macOS` with
  `      torch (CI matrix or a Linux container), not only my local environment`.

- [ ] **Step 6: Build the docs and spell-check.**

  Run: `uv run --group docs mkdocs build --strict 2>&1 | tail -2 && uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes`
  Expected: the build succeeds, and codespell prints nothing.

- [ ] **Step 7: Commit.**
  ```bash
  git add docs/install.md CONTRIBUTING.md .github/PULL_REQUEST_TEMPLATE.md
  git commit -m "docs: Intel macOS is unsupported; 0.14.x on Python 3.10/3.11 is the fallback"
  ```

### Task 6: Changelog fragments

**Files:** `changes/236.removed.md`, plus a new `changes/+drop-intel-mac.removed.md`

- [ ] **Step 1: Remove the Intel sentence from 236.** In
  `changes/236.removed.md`, delete the sentence
  `On Intel macOS, where torch stops at 2.2.x, that leaves 3.11 as the only supported Python. `
  (including its trailing space).

- [ ] **Step 2: The new fragment.** Write `changes/+drop-intel-mac.removed.md`:

  ```markdown
  Intel macOS (x86_64) is no longer supported: PyTorch publishes no x86_64 macOS wheels after torch 2.2, which held mushin there to torch 2.2.x and NumPy 1.x. mushin 0.14.x is the last series that runs on Intel Macs, on Python 3.10 or 3.11 (`python3.11 -m pip install "mushin-py<0.15"`). Apple Silicon Macs remain fully supported and tested.
  ```

- [ ] **Step 3: Preview.**

  Run: `uv run towncrier build --draft --version 0.15.0 2>&1 | sed -n '/### Removed/,/^### [^R]/p'`
  Expected: two Removed bullets, neither mentioning Intel macOS as supported.

- [ ] **Step 4: Commit.**
  ```bash
  git add changes/
  git commit -m "docs(changelog): fragment for dropping Intel macOS"
  ```

### Task 7: Update the 3.15 promotion checklist

**Files:** `docs/superpowers/specs/2026-10-14-python-311-315-design.md`

- [ ] **Step 1: Add the dated note.** Insert this immediately before the line
  `## Non-goals`, with a blank line before and after:

  ```markdown
  **Update 2026-10-14 (Intel macOS dropped; see
  [its spec](2026-10-14-drop-intel-mac-design.md)):**
  - **Criterion 3 / blocker 4.** Without the Intel numpy<2 fork,
    `uv lock --upgrade-package scipy --upgrade-package contourpy` gives scipy
    1.17.1 for 3.11 and 1.18.1 (cp315 wheels) for >= 3.12. contourpy splits the
    same way, 1.3.3 and 1.4.0. No marker split is needed. Do it explicitly in
    the promotion PR: Dependabot cannot make that split.
  - **Dependabot now bumps torch and torchvision.** The promotion PR confirms
    the locked versions have cp315 wheels, or bumps them by hand. numpy (two
    locked versions) is still bumped by hand.
  - **The `dependabot.yml` torch/numpy note is gone.** That note, cited in item
    3, was removed together with the marker split.
  ```

- [ ] **Step 2: Commit.**
  ```bash
  git add docs/superpowers/specs/2026-10-14-python-311-315-design.md
  git commit -m "docs: update the 3.15 promotion checklist for the Intel-mac drop"
  ```

### Task 8: Verification

- [ ] **Step 1: The local gate.**

  Run: `uv run ruff check . && uv run ruff format --check . && uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes && uv run --extra eval --extra detection --extra image --extra audio pytest tests/ --hypothesis-profile fast -p no:cacheprovider -n 6 -q`
  Expected: clean, and `0 failed`.

- [ ] **Step 2: Refusal on Intel.**

  Run:
  ```bash
  R=$(mktemp -d) && git archive HEAD | tar -x -C "$R" && cd "$R" && UV_PROJECT_ENVIRONMENT="$R/v" uv sync --python ~/.local/share/uv/python/cpython-3.11-macos-x86_64-none/bin/python3.11 2>&1 | tail -1; cd - && rm -rf "$R"
  ```
  Expected:
  `error: The current Python platform is not compatible with the lockfile's supported environments: ...`

- [ ] **Step 3: Floors.** Run `make test-lowest`, with Docker healthy and the
  lock checksum taken before and checked after, as in the 3.10 plan's Task 5
  Step 2.
  Expected:
  - `752 passed` (unchanged from #236's run);
  - `uv.lock: OK`.

- [ ] **Step 4: Leftover wording.**

  Run: `git grep -nIiE "intel|x86_64|platform-split|split by platform|marker.split|non-Intel|numpy-1|numpy 1\.x" -- . ':!uv.lock' ':!CHANGELOG.md' ':!docs/superpowers' ':!*.ipynb'`
  Expected, and nothing else:
  - `Makefile:26` (x86_64 Linux);
  - the new "Intel macOS is not supported / unsupported" statements in
    `pyproject.toml`, `ci.yml`, `docs/install.md` and the fragment;
  - the `_provenance.py` Intel-CPU MPS lines and their tests
    (`tests/test_provenance.py`);
  - `mcp/server.py:61` (numpy 1.x unpickling);
  - `batteries.md`/`_metrics.py` "intelligibility" false positives.

### Task 9: PR, Codex, merge

- [ ] **Step 1: Push and open the PR.** Use
  `git push -u origin build/drop-intel-mac`, then `gh pr create --base main`.
  The body covers:
  - why Intel macOS is dropped;
  - the lock scoping;
  - the Dependabot change and its limits;
  - that PESQ was re-tested;
  - the Intel fallback;
  - the verification evidence.

  No Claude attribution.

- [ ] **Step 2: Number the fragment.** Rename it with
  `git mv changes/+drop-intel-mac.removed.md changes/<PR>.removed.md`, commit,
  and push.

- [ ] **Step 3: Wait for Codex and CI.** Codex's 👍 with no inline comments
  means clean. Also wait for `ci-ok`, the `macos-latest` 3.11 leg, and
  `min-versions`. Address real Codex findings.

- [ ] **Step 4: Merge** with `gh pr merge <N> --squash`. Then confirm the
  post-merge `main` CI is green, and push nothing to `main` while it runs.
