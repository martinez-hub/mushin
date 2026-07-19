# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import importlib
from typing import TYPE_CHECKING

from ._packing import pin_gpu_round_robin
from ._resume import ResumeContext
from ._sweep_decorator import sweep
from ._tuning import tune_batch_size, tune_learning_rate
from ._utils import load_experiment, load_from_checkpoint, original_cwd
from .study import (
    Study,  # keep last of eager block: avoids circular import via .study -> _sweep
)
from .workflows import MultiRunMetricsWorkflow, hydra_list, multirun

# The Lightning integration (`HydraDDP`, `MetricsCallback`) is loaded on first
# attribute access (see __getattr__): `mushin.lightning.callbacks` imports
# pytorch_lightning, which alone accounts for ~1.1s (~65%) of a cold `import
# mushin` and is unused by the sweep -> xarray core. `_tuning`/`_study` import
# pytorch_lightning only inside functions, so they stay eager and cheap.
_LAZY_LIGHTNING = frozenset(
    {
        "HydraDDP",
        "HydraFSDP",
        "MetricsCallback",
        "submitit_slurm_config",
        "seed_everything_per_rank",
    }
)

# Benchmark exports are loaded on first attribute access (see __getattr__), so a
# bare `import mushin` does not pull torchmetrics-heavy battery code.
_LAZY_BENCHMARK = frozenset(
    {
        "BenchmarkResult",
        "Task",
        "compare",
        "register_task",
        "get_task",
        "list_tasks",
        "audio_battery",
        "classification_battery",
        "segmentation_battery",
        "detection_battery",
        "regression_battery",
        "retrieval_battery",
        "image_quality_battery",
    }
)

# Legacy names kept importable from the top level for one release, with a warning
# pointing at their new home. They are NOT advertised in __all__.
_DEPRECATED = {
    "BaseWorkflow": "mushin.workflows",
    "RobustnessCurve": "mushin.workflows",
}

if TYPE_CHECKING:  # help static analysers/IDEs see the lazy names
    from . import llm
    from .benchmark import (
        BenchmarkResult,
        Task,
        audio_battery,
        classification_battery,
        compare,
        detection_battery,
        get_task,
        image_quality_battery,
        list_tasks,
        register_task,
        regression_battery,
        retrieval_battery,
        segmentation_battery,
    )
    from .lightning import (  # noqa: F401
        HydraDDP,
        HydraFSDP,
        MetricsCallback,
        seed_everything_per_rank,
        submitit_slurm_config,
    )

    # Not in __all__ (deprecated top-level names); kept for type-checkers only.
    from .workflows import BaseWorkflow, RobustnessCurve  # noqa: F401


def __getattr__(name: str):
    if name == "llm":
        module = importlib.import_module("mushin.llm")
        globals()["llm"] = module
        return module
    if name in _LAZY_BENCHMARK:
        value = getattr(importlib.import_module("mushin.benchmark"), name)
        globals()[name] = value  # cache so later lookups skip __getattr__
        return value
    if name in _LAZY_LIGHTNING:
        value = getattr(importlib.import_module("mushin.lightning"), name)
        globals()[name] = value  # cache so later lookups skip __getattr__
        return value
    if name in _DEPRECATED:
        import warnings

        warnings.warn(
            f"mushin.{name} is deprecated and will be removed in a future "
            f"release; import it from {_DEPRECATED[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Intentionally NOT cached in globals(): re-import is cheap and keeps the
        # warning firing on each top-level access during the deprecation window.
        return getattr(importlib.import_module(_DEPRECATED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_DEPRECATED))


__all__ = [
    "llm",
    "ResumeContext",
    "load_experiment",
    "load_from_checkpoint",
    "original_cwd",
    "sweep",
    "tune_batch_size",
    "tune_learning_rate",
    "MetricsCallback",
    "MultiRunMetricsWorkflow",
    "HydraDDP",
    "pin_gpu_round_robin",
    "HydraFSDP",
    "submitit_slurm_config",
    "seed_everything_per_rank",
    "multirun",
    "hydra_list",
    "Study",
    "compare",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "audio_battery",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
    "regression_battery",
    "retrieval_battery",
    "image_quality_battery",
]
