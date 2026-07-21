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
    # tasks_per_node is derived from gpus_per_node and must not be overridable
    # via **extra (that desync is exactly the footgun this helper prevents)
    with pytest.raises(ValueError, match="tasks_per_node"):
        submitit_slurm_config(nodes=1, gpus_per_node=4, tasks_per_node=1)


def test_seed_everything_per_rank_offsets_by_global_rank(monkeypatch, tmp_path):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.chdir(tmp_path)  # the helper records the seed into the cwd
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert seed_everything_per_rank(1000) == 1003  # base + global rank

    monkeypatch.setenv("RANK", "5")  # RANK takes precedence
    assert seed_everything_per_rank(1000) == 1005


def test_seed_everything_per_rank_defaults_rank_zero(monkeypatch, tmp_path):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.chdir(tmp_path)  # the helper records the seed into the cwd
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
