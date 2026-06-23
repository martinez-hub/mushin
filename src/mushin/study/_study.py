# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The Study class: train-sweep -> compare, plus from_checkpoints eval-only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mushin.benchmark import BenchmarkResult

from ._load import evaluate_checkpoints
from ._sweep import TrainFn, run_training_sweep


class Study:
    """Run a method x seed training sweep and compare the trained models.

    Use the constructor for the full motion (train + compare), or
    ``Study.from_checkpoints`` to compare already-trained checkpoints.
    """

    def _init_common(self, load_fn, data, num_classes, task, test, alpha):
        self._load_fn = load_fn
        self._data = data
        self._num_classes = num_classes
        self._task = task
        self._test = test
        self._alpha = alpha

    def __init__(
        self,
        methods: dict[str, TrainFn],
        load_fn: Callable[[str], Any],
        seeds: Sequence[int],
        data,
        *,
        num_classes: int,
        task: str = "classification",
        test: str = "wilcoxon",
        alpha: float = 0.05,
        working_dir: str | None = None,
    ):
        self._init_common(load_fn, data, num_classes, task, test, alpha)
        self._methods = methods
        self._seeds = list(seeds)
        self.working_dir = working_dir
        self.checkpoints: dict[str, list[str]] | None = None

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: dict[str, Sequence[str]],
        load_fn: Callable[[str], Any],
        data,
        *,
        num_classes: int,
        task: str = "classification",
        test: str = "wilcoxon",
        alpha: float = 0.05,
    ) -> Study:
        """Build a Study that compares already-trained checkpoints (no training)."""
        if not checkpoints:
            raise ValueError("checkpoints must not be empty")
        study = cls.__new__(cls)
        study._init_common(load_fn, data, num_classes, task, test, alpha)
        study._methods = None
        study._seeds = None
        study.working_dir = None
        study.checkpoints = {m: list(p) for m, p in checkpoints.items()}
        return study

    def run(self) -> BenchmarkResult:
        if self._methods is not None:
            ckpt_dir = Path(self.working_dir or ".").resolve() / "study_checkpoints"
            self.checkpoints = run_training_sweep(
                self._methods, self._seeds, ckpt_dir, self.working_dir
            )
        if self.checkpoints is None:
            raise RuntimeError(
                "no checkpoints to evaluate; provide `methods` or use "
                "Study.from_checkpoints(...)"
            )
        return evaluate_checkpoints(
            self.checkpoints,
            self._load_fn,
            self._data,
            self._task,
            self._num_classes,
            self._test,
            self._alpha,
        )
