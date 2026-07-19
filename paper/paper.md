---
title: 'mushin: boilerplate-free, reproducible machine-learning experiment sweeps'
tags:
  - Python
  - machine learning
  - reproducibility
  - experiment management
  - PyTorch
  - hyperparameter sweeps
authors:
  - name: Josue Martinez-Martinez
    orcid: 0000-0000-0000-0000  # TODO: add your ORCID
    affiliation: 1
affiliations:
  - name: TODO — your institution
    index: 1
date: 18 July 2026
bibliography: paper.bib
---

# Summary

`mushin` is a Python library for running reproducible machine-learning
experiment sweeps with minimal boilerplate. A researcher decorates an
experiment function with `@mushin.sweep`, declares the parameters to sweep, and
receives the results as a labeled `xarray.Dataset` [@hoyer2017xarray] keyed by
the swept dimensions — rather than as rows in a dashboard that must be exported
and re-assembled. The sweep layer is framework-agnostic: a task simply returns a
dictionary of metrics, so scikit-learn [@pedregosa2011scikit], XGBoost, or any
Python model can be swept the same way as a PyTorch [@paszke2019pytorch] model.

Built on PyTorch Lightning [@falcon2019lightning] and hydra-zen
[@hydrazen2022] (a typed, programmatic interface to Hydra [@yadan2019hydra]),
`mushin` records each run's configuration and provenance, writes results to
timestamped directories, and aggregates them automatically. The same experiment
code scales from a laptop to a multi-node SLURM cluster by changing only the
launcher, and mushin adds reproducibility features — durable resume after
interruption, hardware-independent auto-tuning, and per-run provenance — that
are otherwise re-implemented ad hoc in each project.

`mushin` is a maintained, standalone extraction of the `mushin` subpackage of
MIT Lincoln Laboratory's `responsible-ai-toolbox` [@raitoolbox2021], whose last
release predates current versions of its dependencies.

# Statement of need

Hyperparameter and seed sweeps are central to empirical machine-learning
research, but the surrounding infrastructure is repetitive and error-prone.
Practitioners commonly stitch together a configuration system, a launcher, a
results store, and analysis glue by hand; the results often end up in an
experiment-tracking dashboard [@biewald2020wandb; @mlflow2018] from which they
must be exported before analysis, and the exact configuration that produced a
number is easy to lose. Hyperparameter-optimization frameworks
[@akiba2019optuna; @liaw2018tune] focus on *searching* for a best configuration
rather than producing the full labeled grid of results that scientific
comparison and ablation require, and they still leave provenance, resumption,
and multi-GPU scaling to the user.

`mushin` targets this gap. Its contribution is a small, composable workflow
layer with four properties that are usually absent together:

- **Sweep → labeled dataset.** The Cartesian product of swept parameters is
  returned directly as an `xarray.Dataset`, so per-seed reduction, slicing, and
  plotting are immediate and the mapping from configuration to result is never
  lost.
- **Reproducibility by construction.** Each run's resolved configuration and
  provenance are captured automatically; auto-tuning helpers pin a tuned batch
  size (with gradient accumulation) so a run is reproducible across hardware;
  and sweeps are durably resumable — a sweep killed mid-run, including by a real
  cluster preemption, resumes without recomputing completed cells.
- **Laptop-to-cluster with one code path.** The same task runs in-process, across
  local worker processes (joblib), or across SLURM nodes (submitit
  [@submitit2021]) by changing only a launcher argument. Data-parallel (DDP) and
  sharded (FSDP) multi-GPU training and round-robin GPU packing for small jobs
  are supported and have been validated on real multi-node GPU hardware.
- **Statistics-aware comparison.** Built-in "batteries" and a `compare` API
  evaluate methods across seeds with significance testing, so claims of
  improvement come with uncertainty rather than a single number.

Because the sweep interface is framework-agnostic and results are standard
`xarray` objects, `mushin` composes with the existing scientific-Python stack
instead of replacing it. It is intended for researchers who want the
reproducibility and scaling machinery of a larger platform without adopting one,
and it is installable from PyPI (`pip install mushin-py`) with documentation,
runnable examples, and a test suite.

# Acknowledgements

`mushin` originates from the `mushin` subpackage of MIT Lincoln Laboratory's
`responsible-ai-toolbox` [@raitoolbox2021]; this package maintains and extends
that workflow layer as a standalone project.

# References
