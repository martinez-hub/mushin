"""Reproducible, statistically-rigorous evaluation of LLM systems."""
# ruff: noqa: E402 - the install-hint guard runs before the imports it protects

# Part of the optional evaluation layer — surface a clear install hint instead of
# a raw scipy/torchmetrics ImportError when the `eval` extra is missing.
from .._extras import require_eval as _require_eval

_require_eval(
    "mushin's LLM evaluation features require the optional 'eval' extra:\n\n"
    "    pip install mushin-py[eval]\n"
)

from ._compare import compare_llms, compare_scores
from ._judge import llm_judge
from ._types import Example, Metric, System

__all__ = ["System", "Metric", "Example", "compare_llms", "compare_scores", "llm_judge"]
