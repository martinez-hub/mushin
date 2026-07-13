# Auto-tuning batch size and learning rate

PyTorch Lightning's `Tuner` can find the largest **device** batch that fits
(`scale_batch_size`) and a learning rate (`lr_find`). Used naively they hurt
reproducibility: the largest device batch depends on the GPU, so the same config
gives different results on a 24 GB vs an 80 GB card, and a sweep silently
compares methods at different batch sizes.

mushin adds two opt-in helpers that keep the convenience while protecting
reproducibility. Both **find once, then pin**: they write the found value to a
small sidecar YAML file and, on a later run, read it and skip the search.

## `tune_batch_size`: pin the effective batch

Pin the **effective** batch (`device_batch x accumulate_grad_batches x
num_devices`) — the hardware-independent, scientifically meaningful quantity.
Call it before `fit`:

```python
from mushin import tune_batch_size

pin = tune_batch_size(trainer, module, datamodule, effective_batch_size=256)
print(pin.device_batch, pin.accumulate_grad_batches, pin.effective_batch_size)
trainer.fit(module, datamodule=datamodule)
```

The helper finds the largest device batch that fits, then reduces it to the
largest value that **divides the per-device target exactly**, so the realized
effective batch always equals your target on any hardware — there is no drift.
The raw hardware probe (the largest batch that fit) is written to
`<trainer.default_root_dir>/mushin_batch_pin.yaml` (override with `pin_path=`);
commit it to make re-runs deterministic. On a later run the probe is read, the
search is skipped, and `device_batch`/accumulation are re-derived for that run's
`effective_batch_size`/`num_devices` — so the same pin works unchanged across
different GPU counts. Pass `retune=True` to search again.

Pick a rounder `effective_batch_size` (256/512/1024 — many divisors) for the best
GPU utilization; a near-prime target may force a small device batch, and the
helper warns when that happens.

Use `num_devices=` if it should not come from the trainer.

## `tune_learning_rate`: record-and-pin the LR finder

```python
from mushin import tune_learning_rate

pin = tune_learning_rate(trainer, module, datamodule)  # sets module.lr
trainer.fit(module, datamodule=datamodule)
```

Learning rate is hardware-independent, so there is no device math — pinning just
makes the stochastic range test skip on re-runs and reuse the exact found value.
The suggestion is written to `<trainer.default_root_dir>/mushin_lr_pin.yaml` and set on
`module.lr` (use `lr_attr=` for a different attribute).

## Caveats

- **Opt-in and explicit.** Both run real training steps and mutate then reset
  trainer/model state — call them deliberately, not on by default.
- **Tune on a single device.** The pinned device batch and recomputed
  accumulation then apply at scale; running the finder itself under DDP is not
  orchestrated for you.
- **Pass an explicit `pin_path` in a sweep.** The default pin directory is shared
  across the jobs of a Hydra `--multirun`, so distinct configs would clobber each
  other's pins. Inside a multirun the helpers require an explicit, per-config
  `pin_path=` (commit it) rather than guess one.
- **Reproducibility is preserved by the pin.** Re-runs reuse the recorded device
  batch regardless of hardware. A pinned batch that no longer fits on smaller
  hardware is a deliberate `retune=True` decision, not a silent change.
