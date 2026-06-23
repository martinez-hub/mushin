# Changelog automation with towncrier — Design

*Date: 2026-06-23*

## Goal

Make every PR document its own user-facing change as a **news fragment**, enforced
by CI, and assemble those fragments into `CHANGELOG.md` mechanically at release
time. This removes the two recurring failure modes of a hand-edited changelog:
(1) contributors forgetting to add an entry, and (2) merge conflicts when several
PRs edit the same `## [Unreleased]` block. Mirrors the workflow contributors
expect from projects like hydra-zen.

Tests already run on every PR (the `test` matrix in `ci.yml`), so this change is
purely about the **changelog gate** — the missing piece.

## Design decisions

| Element | Decision |
| --- | --- |
| Tool | [`towncrier`](https://towncrier.readthedocs.io/), Markdown output (verified to reproduce our Keep-a-Changelog format exactly). |
| Fragment location | `changes/` at repo root; one file per change: `changes/<id>.<type>.md`. |
| Fragment id | PR or issue number (`30.fixed.md`), or `+<slug>` for an orphan with no number (`+lockfile-sync.fixed.md`). |
| Types → sections | `added`→Added, `changed`→Changed, `fixed`→Fixed, `removed`→Removed, `deprecated`→Deprecated (1:1 with our existing Keep-a-Changelog sections), plus `misc` (showcontent=false) as the escape hatch for CI-only / typo / refactor PRs. |
| Traceability | Each rendered line gets a `(#NN)` suffix via `issue_format = "#{issue}"`. |
| Enforcement | New `changelog` job in `ci.yml` on `pull_request`: `towncrier check --compare-with origin/${{ github.base_ref }}`. Fails when no fragment was added. Skipped for Dependabot PRs. |
| Release assembly | Maintainer runs `towncrier build --version X.Y.Z` (wrapped as `make changelog`); it folds fragments into a new `## [X.Y.Z] - DATE` block under a marker and deletes them. Replaces hand-editing the changelog. |
| Skip hatch | A `misc` fragment (e.g. `changes/+ci.misc.md`) — renders only a terse `### Misc / - #NN`, no prose required. No label mechanism. |

## Owned vs. delegated

Delegated: changelog rendering/assembly to towncrier, the CI runner to GitHub
Actions. Owned: the config that pins towncrier to our exact format, the CI job
wiring, and the contributor docs. No application code changes.

## Architecture / components

- **`pyproject.toml`** — add `[tool.towncrier]` (verified config):
  ```toml
  [tool.towncrier]
  directory = "changes"
  filename = "CHANGELOG.md"
  start_string = "<!-- towncrier release notes start -->\n"
  underlines = ["", "", ""]            # Markdown headings, not RST underlines
  title_format = "## [{version}] - {project_date}"
  issue_format = "#{issue}"

  [[tool.towncrier.type]]
  directory = "added"
  name = "Added"
  showcontent = true
  # ... changed / fixed / removed / deprecated identically ...

  [[tool.towncrier.type]]
  directory = "misc"
  name = "Misc"
  showcontent = false
  ```
  Add `towncrier >= 23.11` to the `dev` dependency group (it ships the Markdown
  support this config relies on).

- **`CHANGELOG.md`** — insert the marker line
  `<!-- towncrier release notes start -->` immediately after the
  `## [Unreleased]` header. towncrier inserts each new version block *below* the
  marker and *above* the existing `## [0.2.0]` block, preserving history. The
  existing `[Unreleased]` / version link-reference footer is left untouched.

- **`changes/`** — new directory containing:
  - `.gitkeep` so the empty dir is tracked.
  - `README.md` documenting the fragment format, the type list, and the `+slug`
    orphan convention, with a copy-paste example.

- **`.github/workflows/ci.yml`** — new job:
  ```yaml
  changelog:
    runs-on: ubuntu-latest
    # PRs only (the check diffs the base branch); never on push to main, and
    # exempt Dependabot (lockfile/deps PRs carry no user-facing fragment).
    if: github.event_name == 'pull_request' && github.actor != 'dependabot[bot]'
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0          # towncrier check diffs against the base branch
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
      - run: uv sync
      - name: Changelog fragment check
        run: uv run towncrier check --compare-with origin/${{ github.base_ref }}
  ```
  Note `fetch-depth: 0` — `towncrier check` needs history to diff the base
  branch. The `if:` guard restricts the job to `pull_request` events (it
  references `github.base_ref`, which is empty on push) and exempts Dependabot.

- **`Makefile`** — add a `changelog` target:
  `uv run towncrier build --version $(VERSION)` (maintainer passes
  `VERSION=0.3.0`), plus a `changelog-draft` target
  (`uv run towncrier build --draft --version $(VERSION)`) to preview without
  consuming fragments. Document that `towncrier` reads the date itself.

- **`CONTRIBUTING.md`** — a new "Changelog" subsection under "Submitting changes":
  explain that every PR needs a `changes/<id>.<type>.md` fragment, list the
  types, show an example, and name the `misc` escape hatch. Add a line to the
  numbered submit checklist.

- **`README` / release docs** — a short "Cutting a release" note (in
  CONTRIBUTING or a `RELEASING.md`) documenting the
  `make changelog VERSION=X.Y.Z` → commit → tag → GitHub release flow, so the
  existing `publish.yml` (tag == version) still drives publishing.

- **Seed fragments** — convert the changes already on this branch into fragments
  so the gate is dogfooded and green:
  - `changes/+lockfile-sync.fixed.md` — "Synced `uv.lock`'s recorded project
    version to 0.2.0."
  - `changes/+changelog-automation.added.md` — "Adopted towncrier news fragments
    with a CI gate; contributors now add a `changes/` fragment per PR."

## Data flow

PR author adds `changes/<id>.<type>.md` → CI `changelog` job runs
`towncrier check` against the base branch → passes iff a fragment was added (or
author is Dependabot). At release, maintainer runs `make changelog VERSION=X.Y.Z`
→ towncrier reads all fragments, groups by type, renders Markdown sections under
a new `## [X.Y.Z] - DATE` block beneath the marker, deletes the fragments →
maintainer commits, tags, and publishes as today.

## Error handling

- No fragment on a code PR → `towncrier check` exits non-zero with a message
  naming the missing fragment; CI fails. Contributor adds a fragment (or a
  `misc` one).
- Shallow checkout → `towncrier check` can't diff; mitigated by `fetch-depth: 0`.
- Dependabot PRs (lockfile/deps only, no fragment) → exempted via the job `if:`.

## Testing strategy

This is infrastructure, so "tests" are reproducible verifications rather than
unit tests:

- **Render fidelity (already verified, re-verify in plan):** a throwaway
  `towncrier build` against a copy of our `CHANGELOG.md` produces the exact
  `### Added` / bulleted / `(#NN)` format under the marker, above the prior
  version. (Confirmed during design.)
- **Gate — negative:** push a commit with no fragment and confirm the
  `changelog` CI job fails; **positive:** add a fragment and confirm it passes.
  Validated on this branch's own PR (the seed fragments make it green).
- **`make changelog-draft`:** running it locally prints the assembled section
  without deleting fragments (non-destructive preview).
- **codespell:** ensure the `changes/` fragments and the new CHANGELOG marker
  don't trip codespell (extend the `skip`/word list only if needed).
- **No regression:** existing `lint` and `test` jobs are unchanged and must stay
  green.

## Non-goals

- No change to application code, the test suite, or `publish.yml`.
- Not auto-running `towncrier build` in CI on release — assembly stays a
  deliberate maintainer step (keeps the changelog reviewable before a tag).
- No automatic PR-number backfill — authors name the fragment with the PR number
  (or `+slug` when unknown).
- No "Security" type for now (add later if a security entry is ever needed).
