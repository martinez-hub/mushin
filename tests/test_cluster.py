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


def test_submitit_slurm_config_rejects_bad_inputs():
    from mushin.lightning import submitit_slurm_config

    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=0, gpus_per_node=4)
    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=1, gpus_per_node=0)


def test_seed_everything_per_rank_offsets_by_global_rank(monkeypatch):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert seed_everything_per_rank(1000) == 1003  # base + global rank

    monkeypatch.setenv("RANK", "5")  # RANK takes precedence
    assert seed_everything_per_rank(1000) == 1005


def test_seed_everything_per_rank_defaults_rank_zero(monkeypatch):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert seed_everything_per_rank(42) == 42


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
