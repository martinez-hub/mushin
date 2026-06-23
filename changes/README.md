# News fragments

Every pull request that changes user-facing behavior must add a **news
fragment** here. At release time these are assembled into `CHANGELOG.md` by
[towncrier](https://towncrier.readthedocs.io/) (`make changelog VERSION=X.Y.Z`).

## How to add one

Create a file named `<id>.<type>.md`, where `<id>` is your PR or issue number
(or `+<slug>` if you don't have one yet), and `<type>` is one of:

| type | CHANGELOG section | use for |
| --- | --- | --- |
| `added` | Added | new features |
| `changed` | Changed | changes to existing behavior |
| `fixed` | Fixed | bug fixes |
| `removed` | Removed | removed features |
| `deprecated` | Deprecated | soon-to-be-removed features |
| `misc` | Misc | CI/refactor/typo PRs with nothing user-facing |

The file holds one line of Markdown describing the change, written for users.
For example, `changes/123.fixed.md` containing:

```
`compare` no longer crashes on single-seed inputs.
```

renders as a bullet under **Fixed** at release time, tagged `(#123)`.

A `misc` fragment satisfies the CI gate without adding a changelog line — use it
for CI/refactor/typo PRs with nothing user-facing (e.g. `changes/+ci.misc.md`).
A numbered misc fragment (`changes/42.misc.md`) leaves a terse `- #42` under a
Misc section; an orphan one (`changes/+slug.misc.md`) renders nothing at all.

## Preview

`make changelog-draft VERSION=0.3.0` renders what the next release section will
look like, without consuming the fragments.
