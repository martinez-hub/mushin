# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The optional `eval` extra: the modules it brings, and whether they are here."""

from __future__ import annotations

import importlib.util

# The third-party modules the evaluation layer imports at module level. The
# install-hint guards in `mushin.benchmark` and `mushin.llm` import these, and
# `eval_available` looks them up, so the two cannot drift apart.
EVAL_MODULES = ("scipy", "torchmetrics")


def eval_available() -> bool:
    """Whether the `eval` extra's modules are installed, without importing them.

    `import mushin` calls this to decide what `__all__` and `dir(mushin)` list,
    so it must stay cheap: importing torchmetrics would import torch.
    """
    try:
        return all(importlib.util.find_spec(name) is not None for name in EVAL_MODULES)
    except (ImportError, ValueError):  # a broken or half-imported module
        return False
