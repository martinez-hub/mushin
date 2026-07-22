# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from .._validate import value_check


class MetricsCallback(Callback):
    """Saves validation and test metrics stored in `trainer.callback_metrics`.

    Parameters
    ----------
    save_dir : str, optional (default=".")

    filename : str, optional (default="metrics.pt")
        The base filename used to store metrics.  For `FITTING` the file is prepended
        with "fit_", for `TESTING` with "test_", and for a standalone
        ``trainer.validate(...)`` run with "validate_".

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
        # Standalone `trainer.validate(...)` rounds — kept apart from the
        # fit-stage validation series so a post-hoc validate neither appends
        # onto nor overwrites `fit_<filename>`.
        self.validate_metrics = defaultdict(list)

    def _get_filename(self, stage: str):
        return self.save_dir / f"{stage}_{self.filename}"

    def _record(self, stored, metrics, epoch=None):
        """Append one round of metrics, keeping every series aligned to the same
        length (== number of validation rounds recorded). With one validation
        per epoch (the default) `list index == epoch`; with intra-epoch or
        every-N-epoch validation the saved `epoch` series is the authoritative
        round -> epoch mapping (e.g. `val_check_interval=0.5` records
        `epoch == [0, 0, 1, 1, ...]`). A metric absent this round is padded
        with NaN; a newly-seen metric is backfilled with NaN for prior rounds.
        When `epoch` is given the callback owns the `epoch` series, so a user
        metric literally named `epoch` is skipped (it must not collide with
        the epoch axis)."""
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
            return
        # A standalone `trainer.validate(...)` fires this same hook; route it
        # to its own series/file so it can't masquerade as fit-stage history.
        fn = getattr(getattr(trainer, "state", None), "fn", None)
        if str(getattr(fn, "value", fn)) == "validate":
            stored, stage = self.validate_metrics, "validate"
        else:
            stored, stage = self.val_metrics, "fit"
        self._record(stored, trainer.callback_metrics, pl_module.current_epoch)
        # Under (multi-node) DDP every rank fires this callback; only rank 0 writes
        # so N ranks don't clobber the same file on a shared filesystem.
        if trainer.is_global_zero:
            torch.save(stored, self._get_filename(stage))

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        self._record(self.test_metrics, trainer.callback_metrics)
        if trainer.is_global_zero:
            torch.save(self.test_metrics, self._get_filename("test"))
