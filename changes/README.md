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

The file holds one line of Markdown describing the change, written for users:

```
changes/123.fixed.md
```
```
`compare` no longer crashes on single-seed inputs.
```

A `misc` fragment can be empty — it only records that a PR happened (e.g.
`changes/+ci.misc.md`), satisfying the CI gate without adding a changelog line.

## Preview

`make changelog-draft VERSION=0.3.0` renders what the next release section will
look like, without consuming the fragments.
