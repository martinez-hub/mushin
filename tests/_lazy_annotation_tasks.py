# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tasks whose annotations name an import made only for type checking (#330).

Python 3.14+ evaluates annotations lazily (PEP 649), so these define fine and
evaluating their annotations raises (``NameError``, ``AttributeError`` for a
submodule, ``TypeError`` for a string stand-in). Before 3.14 a definition
evaluates its annotations, so only the 3.14+ tests import this module. The
tasks are module-level so a parallel launcher's workers can unpickle them.
"""

from typing import TYPE_CHECKING

import _lazy_pkg
import mushin
from mushin.workflows import MultiRunMetricsWorkflow

if TYPE_CHECKING:
    from decimal import Decimal
    from fractions import Fraction

    import _lazy_pkg.sub
else:
    # A stand-in some code keeps for its type-checking-only names: evaluating
    # `Fraction | None` then raises TypeError, not NameError.
    Fraction = "Fraction"


def task(a: int, h: Decimal = None) -> dict:
    return {"v": float(a)}


def resumable(a: int, h: Decimal = None, mushin_resume=None) -> dict:
    return {"v": float(a), "has_resume": float(mushin_resume is not None)}


class LazyWF(MultiRunMetricsWorkflow):
    @staticmethod
    def pre_task(a: int, h: Decimal = None) -> None:
        pass

    @staticmethod
    def task(a: int, h: Decimal = None) -> dict:
        return {"v": float(a)}


# A submodule imported only for type checking: evaluating raises AttributeError.
def dotted(a: int, s: _lazy_pkg.sub.T = None) -> dict:
    return {"v": float(a)}


def standin(a: int, f: Fraction | None = None) -> dict:
    return {"v": float(a)}


def with_self(self, a: int, h: Decimal = None) -> dict:
    return {"v": float(self + a)}


def metric(output, reference, seed: int = 0, d: Decimal = None) -> float:
    return 1.0


# A module-level handle, so a parallel launcher unpickles its task by reference.
@mushin.sweep
def swept(a: int, h: Decimal = None) -> dict:
    return {"v": float(a)}
