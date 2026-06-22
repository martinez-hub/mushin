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
- **North star (grow into this):** a unified API for **defining, executing,
  evaluating, and reporting** scientific experiments across heterogeneous
  tooling — with mushin **owning the evaluate + report spine** and inheriting or
  delegating the rest. See "North star & architecture constraint" below. Claimed
  only once the depth backs it.

In one line: **sell the wedge, build on the ecosystem, aim at the category.**

## North star & architecture constraint

**Mission (the destination):** *mushin provides a unified API for defining,
executing, evaluating, and reporting scientific experiments across heterogeneous
tooling.*

This is the MLOps-framework category (ZenML, Metaflow, Kedro, MLflow). What kills
projects here — and what killed the original mushin — is the **integration
treadmill**: a layer that wraps every backend must chase all of them forever. The
only solo-sustainable path depends on one rule: **own a single object; inherit or
delegate everything else.**

**The owned vs. inherited vs. seam boundary:**

- **Define** — *inherited.* hydra-zen already turns Python into reproducible
  configs.
- **Execute across heterogeneous backends** — *inherited.* Hydra's launcher
  plugins already run the same experiment on local / Slurm / Ray / Joblib /
  cloud. mushin does not build this.
- **Evaluate + Report** — *owned.* This is mushin's territory and its moat. The
  results dataset is the substrate; the value is what is built on it.
- **Trackers, evaluators, exporters** — *thin, optional, ideally
  community-maintained seams* around the core dataset. Never god-abstractions
  mushin must maintain.

**The owned territory, bounded** (so "evaluate + report" does not itself balloon
back into a dashboard/storage product):

- *In:* aggregate results across seeds/runs **with statistics** (e.g. mean ± CI);
  **compare** methods/configs and answer "is the difference real?"
  (significance); emit **reproducible, publication-grade artifacts** (LaTeX /
  markdown tables, paper figures) that regenerate from the dataset.
- *Out:* live training dashboards, hosted run history, orchestration, data/model
  versioning. Those stay W&B's / Hydra's / DVC's job.

**One-liner for the owned spine:** *from sweep to publication-ready comparison,
reproducibly.* The public lead message is unchanged — "run a sweep, get a
dataset" is the on-ramp and substrate; evaluate + report is where the depth and
the moat accrue.

**The core unifying object:** "an experiment is a function; its results are a
labeled dataset," made portable across *where* it ran and *what* logged it. The
litmus test for any addition: does it make that object more useful, or does it
drag mushin into owning a seam? If the latter, it belongs in a plugin, not the
core.

**How the first evaluate + report feature gets chosen:** not from this document.
It comes from dogfooding — running a real experiment through mushin and noticing
the first moment work drops into a notebook to hand-compute a comparison or build
a figure. That friction is the first feature, because it is real.

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
