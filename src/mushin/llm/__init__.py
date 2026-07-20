"""Reproducible, statistically-rigorous evaluation of LLM systems."""

# Part of the optional evaluation layer — surface a clear install hint instead of
# a raw torchmetrics ImportError when the `eval` extra is missing.
try:
    import torchmetrics as _torchmetrics  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised by the core-only CI job
    raise ImportError(
        "mushin's LLM evaluation features require the optional 'eval' extra:\n\n"
        "    pip install mushin-py[eval]\n"
    ) from _exc

from ._compare import compare_llms
from ._judge import llm_judge
from ._types import Example, Metric, System

__all__ = ["System", "Metric", "Example", "compare_llms", "llm_judge"]
