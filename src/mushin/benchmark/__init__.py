# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Standard evaluation protocols: run a benchmark, get a labeled dataset back."""

from ._result import BenchmarkResult
from .compare import compare

__all__ = ["compare", "BenchmarkResult"]
