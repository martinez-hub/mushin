# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Submit a sweep out-of-process with a Hydra launcher (parallel / scheduler).

By default a mushin sweep runs its cells in-process, sequentially. Install a Hydra
launcher plugin and pass ``launcher=`` to run the cells across worker processes —
mushin's per-cell dispatch is picklable, so the cells ship to the workers.

Run it (needs ``pip install "hydra-joblib-launcher"``)::

    python examples/parallel_sweep.py
"""

from __future__ import annotations

import mushin


# --8<-- [start:parallel]
@mushin.sweep
def experiment(lr, seed):
    # Your real training/eval goes here; this toy just returns a metric so the
    # example runs fast. Keep the task importable (module-level, as here) so it
    # ships cleanly to a worker process.
    import math
    import random

    base = 1.0 - (math.log10(lr) + 1.0) ** 2 / 4  # peaks at lr=0.1
    noise = 0.02 * (random.Random(seed).random() - 0.5)
    return dict(accuracy=max(0.0, min(1.0, base + noise)))


def run_parallel(working_dir=None, launcher="joblib"):
    """Run the ``lr × seed`` sweep OUT OF PROCESS via a Hydra launcher plugin,
    returning the labeled ``xarray.Dataset``.

    ``launcher="joblib"`` parallelizes across local cores (needs
    ``hydra-joblib-launcher``). For a SLURM cluster, install
    ``hydra-submitit-launcher`` and pass ``launcher="submitit_slurm"`` with the
    scheduler fields, e.g.::

        experiment.run(
            lr=mushin.multirun([...]), seed=mushin.multirun([...]),
            launcher="submitit_slurm",
            launcher_config=mushin.submitit_slurm_config(
                nodes=1, gpus_per_node=1, timeout_min=60
            ),
        )
    """
    return experiment.run(
        lr=mushin.multirun([0.01, 0.1, 1.0]),
        seed=mushin.multirun([0, 1, 2]),
        working_dir=str(working_dir) if working_dir is not None else None,
        launcher=launcher,
    )


# --8<-- [end:parallel]


def main() -> None:
    ds = run_parallel()
    print(ds)
    print("\nmean accuracy by learning rate:")
    print(ds["accuracy"].mean("seed"))


if __name__ == "__main__":
    main()
