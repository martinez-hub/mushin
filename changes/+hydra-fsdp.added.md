`HydraFSDP`: a Fully-Sharded Data Parallel strategy that works under Hydra
`--multirun`. Like `HydraDDP`, it reattaches ranks via the job's saved
`config.yaml` instead of re-executing with `sys.argv` (which a sweep would run as
the wrong job), so FSDP sharded training composes with Hydra sweeps. Exported from
`mushin`; see the new "Sharded training under Hydra" guide.
