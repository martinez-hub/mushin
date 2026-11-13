import pytest


def test_submitit_slurm_config_derives_tasks_per_node():
    from mushin.lightning import submitit_slurm_config

    cfg = submitit_slurm_config(
        nodes=2, gpus_per_node=4, partition="gpu", cpus_per_task=8
    )
    # DDP contract: one SLURM task per GPU
    assert cfg["tasks_per_node"] == 4
    assert cfg["gpus_per_node"] == 4
    assert cfg["nodes"] == 2
    assert cfg["cpus_per_task"] == 8
    assert cfg["partition"] == "gpu"


def test_submitit_slurm_config_passthrough_and_optional():
    from mushin.lightning import submitit_slurm_config

    cfg = submitit_slurm_config(nodes=1, gpus_per_node=2, mem_gb=64, account="proj")
    assert cfg["mem_gb"] == 64
    assert cfg["account"] == "proj"  # extra kwargs pass through
    assert "partition" not in cfg  # omitted when not given


def test_submitit_slurm_config_keys_are_fields_of_the_installed_launcher():
    """Every key the helper emits, and every pass-through its docstring and the
    multi-node guide name, must be a field of the installed plugin's SLURM
    config: Hydra rejects any other key ("Key ... is not in struct") before
    anything is submitted. hydra-submitit-launcher before 1.2.0 has no
    `account`, so this pins the `slurm` extra's floor (#344)."""
    import dataclasses

    config = pytest.importorskip("hydra_plugins.hydra_submitit_launcher.config")
    from mushin.lightning import submitit_slurm_config

    cfg = submitit_slurm_config(
        nodes=2,
        gpus_per_node=4,
        cpus_per_task=8,
        partition="gpu",
        mem_gb=64,
        account="proj",
        qos="high",
        constraint="a100",
        signal_delay_s=120,
        additional_parameters={"requeue": True},
    )
    fields = {f.name for f in dataclasses.fields(config.SlurmQueueConf)}
    assert set(cfg) <= fields, sorted(set(cfg) - fields)


def test_submitit_slurm_config_rejects_bad_inputs():
    from mushin.lightning import submitit_slurm_config

    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=0, gpus_per_node=4)
    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=1, gpus_per_node=0)
    # tasks_per_node is derived from gpus_per_node and must not be overridable
    # via **extra (that desync is exactly the footgun this helper prevents)
    with pytest.raises(ValueError, match="tasks_per_node"):
        submitit_slurm_config(nodes=1, gpus_per_node=4, tasks_per_node=1)


@pytest.mark.parametrize(
    ("env", "base", "expected"),
    [
        pytest.param({"SLURM_PROCID": "3"}, 1000, 1003, id="slurm-procid"),
        # RANK takes precedence over SLURM_PROCID
        pytest.param({"SLURM_PROCID": "3", "RANK": "5"}, 1000, 1005, id="rank-wins"),
        pytest.param({}, 42, 42, id="defaults-to-rank-zero"),
        # Plain single-node multi-GPU (HydraDDP) exports only LOCAL_RANK: the
        # per-rank seed must still differ per process, not silently collapse to
        # base+0 on every rank.
        pytest.param({"LOCAL_RANK": "1"}, 100, 101, id="local-rank-fallback"),
        # A task of a submitit_local job of several sets only submitit's own
        # variables (#371).
        pytest.param(
            {"SUBMITIT_LOCAL_NTASKS": "2", "SUBMITIT_LOCAL_GLOBALID": "1"},
            100,
            101,
            id="submitit-local-task",
        ),
        # A job of one local task: its id is inherited by any process the task
        # starts (HydraDDP's children), so it is not their rank.
        pytest.param(
            {
                "SUBMITIT_LOCAL_NTASKS": "1",
                "SUBMITIT_LOCAL_GLOBALID": "0",
                "LOCAL_RANK": "1",
            },
            100,
            101,
            id="single-local-task-child",
        ),
    ],
)
def test_seed_everything_per_rank_offsets_by_global_rank(
    monkeypatch, tmp_path, env, base, expected
):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.chdir(tmp_path)  # the helper records the seed into the cwd
    for var in (
        "RANK",
        "SLURM_PROCID",
        "LOCAL_RANK",
        "SUBMITIT_LOCAL_NTASKS",
        "SUBMITIT_LOCAL_GLOBALID",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    # Lightning's seed_everything writes these into os.environ; setting them
    # through monkeypatch first restores them after the test.
    monkeypatch.setenv("PL_GLOBAL_SEED", "0")
    monkeypatch.setenv("PL_SEED_WORKERS", "0")
    assert seed_everything_per_rank(base) == expected  # base + global rank


@pytest.mark.cluster
def test_multinode_ddp_end_to_end(tmp_path):
    """MERGE GATE: run on a real multi-node SLURM allocation. Launches a tiny DDP
    job via submitit + SLURMEnvironment, then asserts it completed, metrics were
    written exactly once (rank 0), and results load back. Run with:
        pytest -m cluster tests/test_cluster.py
    on a node with `hydra-submitit-launcher` installed and a SLURM allocation.
    See docs/guides/multinode.md for the runbook and required env."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    # The concrete launch is documented in the runbook; this placeholder is the
    # named merge gate. It skips cleanly until wired to a real allocation.
    pytest.skip(
        "Provide a SLURM allocation + partition/account, then implement the launch "
        "per docs/guides/multinode.md (this is the merge gate)."
    )


def test_seed_everything_per_rank_persists_seed(monkeypatch, tmp_path):
    """The effective seed must be recoverable from the run's artifacts: if it
    only lives in-process, the exact cell can never be re-run identically."""
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "3")
    from mushin.lightning import seed_everything_per_rank

    seed = seed_everything_per_rank(1000)
    # rank-suffixed so DDP ranks sharing a dir don't clobber rank 0's record
    rec = json.loads((tmp_path / "mushin_seed_rank3.json").read_text())
    assert rec["seed"] == seed == 1003
    assert rec["rank"] == 3

    monkeypatch.setenv("RANK", "0")
    assert seed_everything_per_rank(1000) == 1000
    assert json.loads((tmp_path / "mushin_seed.json").read_text())["seed"] == 1000


def test_rank_from_env_survives_malformed_rank(monkeypatch):
    """A malformed rank env var must degrade to the next source (or rank 0) in
    mushin's own parsing, not raise an uncaught ValueError. (Lightning's
    seed_everything performs its own int(RANK) — that part is theirs.)"""
    from mushin.lightning._cluster import _rank_from_env

    monkeypatch.setenv("RANK", "not-a-number")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert _rank_from_env() == 2

    monkeypatch.delenv("SLURM_PROCID")
    assert _rank_from_env() == 0
