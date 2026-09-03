# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""End-to-end coverage for the DDP child entry point, WITHOUT needing GPUs.

`HydraDDP` re-launches every rank beyond the first as a fresh interpreter:

    python -m mushin.lightning._pl_main -cp <job>/.hydra -cn config.yaml ...

That child re-composes the config from disk, re-configures Hydra job logging,
and re-instantiates the Trainer/module/datamodule. It is the part of the DDP
path most exposed to an upstream Hydra change — instantiate authorization and
logging configuration are both re-run there, in an interpreter that never saw
the parent process.

Nothing exercised it. `tests/test_lightning_hydra_ddp.py` skips the whole module
below 2 GPUs, and `tests/test_lightning_launchers.py` monkeypatches
`_subprocess_call` away, so on every CPU machine and in CI the child entry point
was covered only by the fact that it imports. These tests close that gap: they
build a real job directory, then run the real command line `_subprocess_call`
constructs, as a real subprocess, on CPU.

Surfaced by an adversarial review of the hydra-core 1.3.5 -> 1.3.6 bump (#202),
which hardened `instantiate` authorization and applied Hydra's target blocklist
to logging configuration. Neither broke mushin — but nothing here would have
noticed if they had.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra_zen import builds, launch, make_config
from pytorch_lightning import Trainer

from mushin.testing.lightning import SimpleDataModule, SimpleLightningModule

# CPU-only and deliberately tiny: `fast_dev_run` runs a single batch through
# each requested loop, which is all that is needed to prove the child composed,
# instantiated and executed.
TrainerConf = builds(
    Trainer,
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    fast_dev_run=True,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
)
Config = make_config(
    trainer=TrainerConf,
    module=builds(SimpleLightningModule),
    datamodule=builds(SimpleDataModule),
)


def _noop(cfg) -> None:
    """The parent job body. The child interpreter is what is under test."""


def _job_dir(tmp_path: Path) -> Path:
    """Run a real Hydra job so a real `.hydra/config.yaml` lands on disk.

    The child is launched from the SAVED config, not from an in-memory object,
    so the config has to make the round trip through YAML for this to test what
    HydraDDP actually does.
    """
    out = tmp_path / "job"
    launch(
        Config,
        _noop,  # the parent does nothing; the child is under test
        [f"hydra.run.dir={out}"],
        version_base="1.3",
        with_log_configuration=True,
    )
    # NB: `job.working_dir` is the ORIGINAL cwd, not the job dir — under
    # version_base >= 1.2 Hydra no longer chdirs into it. `launchers.py` resolves
    # this from HydraConfig for the same reason; here we simply know the path,
    # because we set `hydra.run.dir` above.
    assert (out / ".hydra" / "config.yaml").is_file(), "no saved config to launch from"
    return out


def _run_child(job: Path, tmp_path: Path, *, entry: str = "fit"):
    """Invoke the child exactly as `launchers._subprocess_call` builds it."""
    flags = {
        "fit": ("false", "false", "false"),
        "test": ("true", "false", "false"),
        "predict": ("false", "true", "false"),
        "validate": ("false", "false", "true"),
    }[entry]
    command = [
        sys.executable,
        "-m",
        "mushin.lightning._pl_main",
        "-cp",
        str(job / ".hydra"),
        "-cn",
        "config.yaml",
        f"++pl_testing={flags[0]}",
        f"++pl_predicting={flags[1]}",
        f"++pl_validating={flags[2]}",
        "++pl_local_rank=0",
        f"hydra.run.dir={tmp_path / ('child_' + entry)}",
        "hydra.output_subdir=.pl_hydra_rank_0",
        "hydra.job.name=pl_main_cpu",
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=600)


def test_child_entry_point_runs_fit_on_cpu(tmp_path):
    """The whole point: a fresh interpreter re-composes and trains from disk."""
    proc = _run_child(_job_dir(tmp_path), tmp_path)
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-3000:]}"
    # This line is emitted through Hydra's job logging, so seeing it proves BOTH
    # that the fit branch ran and that `configure_log` succeeded in the child —
    # the logging path hydra 1.3.6's blocklist change touches.
    assert "Launched subprocess using Training.fit" in (proc.stdout + proc.stderr)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("test", "Launched subprocess using Training.test"),
        ("predict", "Launched subprocess using Trainer.predict"),
        ("validate", "Launched subprocess using Trainer.validate"),
    ],
)
def test_child_entry_point_selects_the_right_trainer_loop(tmp_path, entry, expected):
    """`pl_testing`/`pl_predicting`/`pl_validating` pick the Trainer entry point.

    Getting this plumbing wrong would silently *train* on a rank that was asked
    to evaluate — a wrong result rather than a crash, and invisible without
    running the real child.
    """
    proc = _run_child(_job_dir(tmp_path), tmp_path, entry=entry)
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-3000:]}"
    assert expected in (proc.stdout + proc.stderr)


def test_child_writes_its_own_rank_scoped_hydra_dir(tmp_path):
    """Ranks must not overwrite each other's Hydra output.

    `_subprocess_call` passes `hydra.output_subdir=.pl_hydra_rank_<n>` for
    exactly this reason; if the child ignored it, every rank would write to
    `.hydra/` and clobber the parent's saved config.
    """
    proc = _run_child(_job_dir(tmp_path), tmp_path)
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-3000:]}"
    child_dir = tmp_path / "child_fit"
    assert (child_dir / ".pl_hydra_rank_0" / "config.yaml").is_file()
    assert not (child_dir / ".hydra").exists(), "child clobbered the default subdir"
