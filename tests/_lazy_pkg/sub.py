# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Never imported at run time: `_lazy_annotation_tasks` names it only for
type checking, so evaluating `_lazy_pkg.sub.T` raises AttributeError."""


class T:
    pass
