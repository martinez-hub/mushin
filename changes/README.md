# News fragments

Every pull request that changes user-facing behavior must add a **news
fragment** here. At release time these are assembled into `CHANGELOG.md` by
[towncrier](https://towncrier.readthedocs.io/) (`make changelog VERSION=X.Y.Z`).

## How to add one

Create a file named `<id>.<type>.md`, where `<id>` is your PR or issue number
(or `+<slug>` if you don't have one yet), and `<type>` is one of the types
below. A `+<slug>` fragment needs no renaming once the PR exists: when a
release is prepared, it is renamed after the PR whose squash merge added it
(the `(#123)` that ends the commit subject).

| type | CHANGELOG section | use for |
| --- | --- | --- |
| `added` | Added | new features |
| `changed` | Changed | changes to existing behavior |
| `fixed` | Fixed | bug fixes |
| `removed` | Removed | removed features |
| `deprecated` | Deprecated | soon-to-be-removed features |
| `misc` | Misc | CI/refactor/typo PRs with nothing user-facing |

The types also pick the next version when a release is prepared with the
default `auto`: any `added`, `changed`, `removed` or `deprecated` fragment
makes it a minor release, and only `fixed` and `misc` fragments a patch
release (`RELEASING.md`).

The file holds one line of Markdown describing the change, written for users.
For example, `changes/123.fixed.md` containing:

```
`compare` no longer crashes on single-seed inputs.
```

renders as a bullet under **Fixed** at release time, tagged `(#123)`.

A `misc` fragment is the escape hatch for PRs with nothing user-facing (CI,
refactors, typos): it satisfies the gate but adds no substantive entry. At
release it renders only a terse line under a **Misc** section — prefer a numbered
`changes/42.misc.md`, which renders `- #42`. A `+slug.misc.md` gets its number
at release time like any other. (One that never gets a number, because no PR
merge added it, shows its *text* instead, even though `misc` is configured
`showcontent = false`, so keep misc fragments empty. The maintainer can drop
the Misc section when cutting the release.)

## Preview

`make changelog-draft VERSION=0.3.0` renders what the next release section will
look like, without consuming the fragments.
