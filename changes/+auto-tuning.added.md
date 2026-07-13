`tune_batch_size` / `tune_learning_rate`: opt-in, reproducibility-preserving
auto-tuning. Lightning's batch/LR finder runs once, the result is pinned to a
sidecar YAML, and later runs reuse it. `tune_batch_size` pins a hardware-
independent effective batch, choosing the largest device batch that both fits and
divides the per-device target exactly, so the effective batch is identical on any
GPU count with no drift.
