`DistributedTeardown` callback: destroys the `torch.distributed` process group at
the end of each Trainer run so consecutive Hydra `--multirun` jobs work with
sharded strategies (`FSDPStrategy`/`DeepSpeedStrategy`), which do not clean it up
themselves. New "Sharded training (FSDP / DeepSpeed)" guide shows configuring
sharding via hydra-zen; mushin's analysis layer is unchanged.
