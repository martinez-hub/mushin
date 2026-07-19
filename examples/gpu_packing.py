# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Pack several small sweep jobs onto each GPU with ``pin_gpu_round_robin``.

The default is one job per GPU — wasteful when each job uses a fraction of a
device. ``pin_gpu_round_robin`` pins each job to one GPU round-robin (via
``CUDA_VISIBLE_DEVICES``) so a parallel launcher can run more jobs than you have
GPUs. Here 4 jobs share 2 GPUs (jobs 0/2 -> GPU 0, jobs 1/3 -> GPU 1).

Requires >=2 GPUs and ``hydra-joblib-launcher`` (each job needs its own fresh
process — pinning must happen before CUDA initializes). Not run in CI.

Run it:
  pip install hydra-joblib-launcher
  python examples/gpu_packing.py

See the GPU-packing guide for sizing ``jobs_per_gpu`` and the Ray alternative
for true fractional-GPU sharing.
"""

from __future__ import annotations

import os

import mushin
from mushin import pin_gpu_round_robin
from mushin.workflows import MultiRunMetricsWorkflow

NUM_GPUS = 2


class PackedSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        # Pin BEFORE any CUDA use. job_index defaults to the Hydra multirun index.
        gpu = pin_gpu_round_robin(num_gpus=NUM_GPUS)
        import torch

        assert torch.cuda.is_available()
        # A real task would build Trainer(devices=1, accelerator="gpu") and train.
        return dict(
            pinned_gpu=float(gpu),
            cuda_visible=float(int(os.environ["CUDA_VISIBLE_DEVICES"])),
        )


if __name__ == "__main__":
    wf = PackedSweep()
    wf.run(
        seed=mushin.multirun([0, 1, 2, 3]),
        working_dir="gpu_packing_runs",
        launcher="joblib",
        # num_gpus * jobs_per_gpu concurrent jobs (2 GPUs x 2 jobs/GPU = 4).
        overrides=["hydra.launcher.n_jobs=4"],
    )
    # Each job pinned round-robin: expect pinned_gpu == [0, 1, 0, 1] across seeds.
    print("DONE", wf.to_xarray()["pinned_gpu"].values.tolist())
