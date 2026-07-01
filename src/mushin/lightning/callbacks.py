# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from .._validate import value_check
from .launchers import _teardown


class MetricsCallback(Callback):
    """Saves validation and test metrics stored in `trainer.callback_metrics`.

    Parameters
    ----------
    save_dir : str, optional (default=".")

    filename : str, optional (default="metrics.pt")
        The base filename used to store metrics.  For `FITTING` the file is prepended
        with "fit_" and for `TESTING` the file is prepended with `test_`.

    Notes
    -----
    No metrics will be saved during `FITTING` if no validation metrics are calculated.
    This is a limitation of PyTorch Lightning. Future versions will save the training
    step metrics when no validation metrics are calculated.

    Examples
    --------
    >>> from pytorch_lightning import Trainer
    >>> from mushin import MetricsCallback

    >>> metrics_callback = MetricsCallback()
    >>> trainer = Trainer(callbacks=[metrics_callback])
    """

    def __init__(
        self,
        save_dir: Path | str = ".",
        filename: Path | str = "metrics.pt",
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = value_check("filename", filename, type_=(str, Path))
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    def _get_filename(self, stage: str):
        return self.save_dir / f"{stage}_{self.filename}"

    def _record(self, stored, metrics, epoch=None):
        """Append one round of metrics, keeping every series aligned to the same
        length (== number of rounds recorded) so `list index == epoch` holds. A
        metric absent this round is padded with NaN; a newly-seen metric is
        backfilled with NaN for prior rounds. When `epoch` is given the callback
        owns the `epoch` series, so a user metric literally named `epoch` is
        skipped (it must not collide with the epoch axis)."""
        incoming = {}
        if epoch is not None:
            incoming["epoch"] = epoch
        for k, v in metrics.items():
            if epoch is not None and k == "epoch":
                continue  # reserved: the callback owns the epoch axis
            if isinstance(v, torch.Tensor):
                v = v.item() if v.ndim == 0 else v.cpu().numpy()
            incoming[k] = v

        n_rounds = max((len(s) for s in stored.values()), default=0)
        for k in set(stored) | set(incoming):
            series = stored[k]
            while len(series) < n_rounds:  # backfill a newly-seen key
                series.append(float("nan"))
            series.append(incoming.get(k, float("nan")))  # value, or NaN if absent

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # Make sure PL is not doing its sanity check run
        if trainer.sanity_checking:
            return self.val_metrics
        self._record(
            self.val_metrics, trainer.callback_metrics, pl_module.current_epoch
        )
        torch.save(self.val_metrics, self._get_filename("fit"))
        return self.val_metrics

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        self._record(self.test_metrics, trainer.callback_metrics)
        torch.save(self.test_metrics, self._get_filename("test"))
        return self.test_metrics


class DistributedTeardown(Callback):
    """Destroy the ``torch.distributed`` process group at the end of each Trainer
    run so consecutive Hydra ``--multirun`` jobs (run in one process) start clean.

    Lightning's ``FSDPStrategy``/``DeepSpeedStrategy`` do not destroy the process
    group on teardown, so without this the next multirun job's
    ``init_process_group`` fails. ``HydraDDP`` does not need it (it clears any
    leftover group at the next job's setup); use this with sharded strategies (or
    any strategy) under ``--multirun``. Do not add it alongside ``HydraDDP`` --
    ``HydraDDP`` performs its own teardown, so combining them is redundant.
    Idempotent and safe in single-process / CPU runs (a no-op when no group is
    initialized).
    """

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        # Pop leaked LOCAL_RANK/NODE_RANK/.../PL_GLOBAL_SEED so the next multirun
        # job re-initializes from a clean environment.
        _teardown()
