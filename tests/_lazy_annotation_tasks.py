# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tasks whose annotations name an import made only for type checking (#330).

Python 3.14+ evaluates annotations lazily (PEP 649), so these define fine and
evaluating their annotations raises ``NameError``. Before 3.14 a definition
evaluates its annotations, so only the 3.14+ tests import this module. The
tasks are module-level so a parallel launcher's workers can unpickle them.
"""

from typing import TYPE_CHECKING

from mushin.workflows import MultiRunMetricsWorkflow

if TYPE_CHECKING:
    from decimal import Decimal


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
