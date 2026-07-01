# Packing small jobs onto GPUs

When you sweep (Hydra `--multirun`) over **small** models that each fit
comfortably on one GPU, the default is **one job per GPU** — a model using ~10%
of a device still occupies the whole thing, wasting most of the cluster. Packing
runs **several small jobs per GPU** to use each device's full potential.

mushin does not assign GPUs — placement is decided by your **launcher** and by
Lightning/torch inside each job. So packing is a launcher/environment recipe.

## joblib / basic launcher: `pin_gpu_round_robin`

Pin each job to one GPU round-robin, and run more jobs concurrently than you have
GPUs. `pin_gpu_round_robin` sets `CUDA_VISIBLE_DEVICES` from the Hydra job index —
call it at the **top** of your task function, before any CUDA use:

```python
from mushin import pin_gpu_round_robin

def task(cfg):
    pin_gpu_round_robin(num_gpus=4)   # this job -> GPU (job_num % 4)
    # ... build the Trainer(devices=1, accelerator="gpu") and train ...
```

Then run the sweep with the joblib launcher and enough concurrency to place
`jobs_per_gpu` jobs on each GPU (here 4 GPUs x 3 jobs/GPU = 12 concurrent):

```bash
python train.py --multirun \
    hydra/launcher=joblib hydra.launcher.n_jobs=12 \
    model=a,b,c,d,e,f,g,h,i,j,k,l
```

`pin_gpu_round_robin` only maps a job to a device; the concurrency
(`n_jobs = num_gpus * jobs_per_gpu`) is your launcher setting.

## Ray: true fractional-GPU sharing (recommended for heavier sharing)

`hydra-ray-launcher` supports fractional GPUs natively — no mushin code needed:

```bash
python train.py --multirun hydra/launcher=ray \
    hydra.launcher.ray.remote.num_gpus=0.25   # 4 jobs share one GPU
```

Ray schedules the fractions for you; use it when you want many jobs truly
time-slicing a device rather than each pinned to a whole one.

## MPS and MIG

- **NVIDIA MPS** (Multi-Process Service) improves compute overlap when several
  small processes share a GPU — start `nvidia-cuda-mps-control -d` before the
  sweep. Combine with the round-robin pinning above.
- **MIG** (A100/H100) partitions one physical GPU into isolated instances; assign
  jobs to slices via `CUDA_VISIBLE_DEVICES=MIG-<uuid>` (MIG instances appear as
  devices), which `pin_gpu_round_robin` does not compute for you — set it directly.

## Caveats

- **Memory / compute contention:** co-located jobs share the device's memory and
  compute. Tune `jobs_per_gpu` down if you hit OOM.
- **Single-GPU-per-job only:** packing is for sweeps where each job uses one GPU.
  It is mutually exclusive with a job that itself claims multiple GPUs
  (`HydraDDP`, FSDP).
- **Reproducibility:** packing changes only *scheduling*, not results — a packed
  sweep produces the same numbers as one-job-per-GPU.
