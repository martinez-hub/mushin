# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The Study class: train-sweep -> compare, plus from_checkpoints eval-only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mushin.benchmark import BenchmarkResult

from ._load import evaluate_checkpoints
from ._sweep import TrainFn, run_training_sweep


class Study:
    """Run a method x seed training sweep and compare the trained models.

    Use the constructor for the full motion (train + compare), or
    ``Study.from_checkpoints`` to compare already-trained checkpoints.
    """

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
        self._methods = methods
        self._load_fn = load_fn
        self._seeds = list(seeds)
        self._data = data
        self._num_classes = num_classes
        self._task = task
        self._test = test
        self._alpha = alpha
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
        study = cls.__new__(cls)
        study._methods = None
        study._load_fn = load_fn
        study._seeds = None
        study._data = data
        study._num_classes = num_classes
        study._task = task
        study._test = test
        study._alpha = alpha
        study.working_dir = None
        study.checkpoints = {m: list(p) for m, p in checkpoints.items()}
        return study

    def run(self) -> BenchmarkResult:
        if self._methods is not None:
            from pathlib import Path

            ckpt_dir = Path(self.working_dir or ".").resolve() / "study_checkpoints"
            self.checkpoints = run_training_sweep(
                self._methods, self._seeds, ckpt_dir, self.working_dir
            )
        assert self.checkpoints is not None
        return evaluate_checkpoints(
            self.checkpoints,
            self._load_fn,
            self._data,
            self._task,
            self._num_classes,
            self._test,
            self._alpha,
        )
