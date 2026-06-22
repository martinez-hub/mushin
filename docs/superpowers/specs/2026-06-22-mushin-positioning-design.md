# mushin — Positioning Design

*Date: 2026-06-22*

## Goal

Make mushin a widely-adopted tool for reproducible deep-learning research
experiments — ultimately "the standard." This document defines the positioning
that work should ladder up to. "Standard" is an outcome to be earned, not a
status to declare; this is the wedge we earn it from.

## Decisions

These were settled through structured Q&A:

| Element | Decision |
| --- | --- |
| Target user (beachhead) | The individual ML researcher (PhD student / applied scientist) running many-config experiments. |
| Adoption motion | Bottom-up. Win the individual, spread to teams → labs organically (how pytest, Lightning, Hydra became standards). All four personas — individual, team, lab, infra engineer — are the end state; the individual is the entry point. |
| Core pain | Running sweeps and then wrangling the results is glue-code hell. |
| Mechanism | Run a sweep, get the results back as a labeled `xarray` dataset. |
| Competitive frame | Experiment trackers (W&B, MLflow). Contrast: "dataset, not dashboard." |
| Scope | Complement, not replacement. mushin is the analysis layer; trackers keep doing logging/history/collaboration. |

## Positioning statement

> For **ML researchers running many-config experiments**, who today **sweep with
> their tooling and then hand-write glob/parse/concat glue or export CSVs from a
> dashboard**, **mushin** is the **experiment-analysis layer** that **hands a
> finished sweep back as a labeled `xarray` dataset you compute on in your own
> Python session**. Unlike **experiment trackers (W&B, MLflow)**, mushin gives
> you **a dataset, not a dashboard** — results as data, ready to slice, reduce,
> and plot. Built on the **hydra-zen + PyTorch Lightning** ecosystem.

**Tagline:** *Run a sweep, get a dataset.*
**Support line:** *Dataset, not dashboard.*

## Messaging hierarchy (the layered synthesis)

The strategy uses three framings at different layers rather than betting on one:

- **Lead message (sell this now):** "Run a sweep, get a dataset — not a
  dashboard." True today, demos in ~10 lines, no rip-and-replace.
- **Architecture & go-to-market (build on this):** the hydra-zen ecosystem —
  reproducible configs, no YAML hand-wrangling. Coordinate with the (active)
  hydra-zen maintainers so mushin reads as a legitimate ecosystem layer, not
  abandonware under a new name.
- **North star (grow into this):** "array-native experimentation" — define your
  experiment as a function, get your results as arrays. The standard-sized
  ambition, claimed only once the depth backs it.

In one line: **sell the wedge, build on the ecosystem, aim at the category.**

## What mushin is explicitly NOT

This anti-scope is load-bearing — it is what keeps the positioning honest and
prevents the over-promise that destroys credibility.

- **Not a tracker.** No live logging, hosted run history, or collaboration UI.
  It coexists with W&B / MLflow.
- **Not a hyperparameter optimizer.** It analyzes the whole result surface, not
  the argmax. (Contrast with Optuna / Ray Tune.)
- **Not a trainer.** Lightning still trains; mushin orchestrates the sweep and
  aggregates the results.
- **Not (yet) a full reproducibility system.** It captures *configs* via
  hydra-zen, not environment, data, or hardware. We do not claim env/data/
  hardware capture until it is built.

## What must be true to earn the position

Sequenced. Each item is a proof point the positioning depends on.

1. **Killer ~10-line example** — a real sweep → one labeled dataset → a plot.
   The single most important artifact; it *is* the pitch.
2. **Dogfood it** — use mushin on a real research workflow of the maintainer's
   own at least once. Credibility precedes evangelism.
3. **Coordinate with hydra-zen maintainers** — outreach to test the
   coordinate-vs-compete path and earn ecosystem legitimacy.
4. **Docs built around "results as data"** — a narrative/tutorial site, not an
   API dump.
5. **Reset version to 0.1.0** — an honest "new project, fresh lineage" signal
   (the current 0.4.0 is inherited from the upstream lineage).
6. **Publish to PyPI and claim the name** — only after the above. The name is
   confirmed available.

## Non-goals for this document

This defines positioning, not the roadmap detail. Turning the six proof points
into a concrete, sequenced implementation plan is the next step.
