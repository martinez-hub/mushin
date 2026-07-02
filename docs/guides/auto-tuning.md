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
The helper finds the largest device batch that fits, then sets
`accumulate_grad_batches` to reach your target. Call it before `fit`:

```python
from mushin import tune_batch_size

pin = tune_batch_size(trainer, module, datamodule, effective_batch_size=256)
print(pin.device_batch, pin.accumulate_grad_batches, pin.effective_batch_size)
trainer.fit(module, datamodule=datamodule)
```

The device batch is maximized for throughput, so the realized effective batch
can differ slightly from the target when the device batch doesn't divide it
evenly — the helper **records the actual value** in `pin.effective_batch_size`
and **warns** when it drifts. The found device batch is written to
`<trainer.default_root_dir>/mushin_batch_pin.yaml` (override with `pin_path=`); commit it
to make re-runs deterministic. Pass `retune=True` to search again — for example
when you deliberately move to hardware where the pinned batch no longer fits.

Use `safety_margin=` (e.g. `0.1`) to back the found maximum off from the OOM
edge, and `num_devices=` if it should not come from the trainer.

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
- **Reproducibility is preserved by the pin.** Re-runs reuse the recorded device
  batch regardless of hardware. A pinned batch that no longer fits on smaller
  hardware is a deliberate `retune=True` decision, not a silent change.
