# The evaluation layer (compare, batteries, LLM eval, Study) is an optional
# `eval` extra. When it isn't installed, skip these tests instead of erroring at
# import — mirrors how the core-only CI job runs.
try:
    import torchmetrics  # noqa: F401
except ImportError:
    collect_ignore_glob = ["*.py"]
