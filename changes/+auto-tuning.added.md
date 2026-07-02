`tune_batch_size` and `tune_learning_rate`: opt-in, reproducibility-preserving
auto-tuning helpers. `tune_batch_size` pins the effective batch, finds the
largest device batch that fits, and sets `accumulate_grad_batches` to reach it
(recording any drift); `tune_learning_rate` records-and-pins Lightning's LR
finder. Both write the found value to a sidecar YAML so re-runs skip the search
and stay deterministic across hardware. New "Auto-tuning batch size and learning
rate" guide.
