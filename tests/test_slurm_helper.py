# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.slurm`: a sweep's SLURM resources, one cell per job, and the
check of the request against `scontrol` (#346). No test reaches a real
`scontrol`: each stands in its output."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

import mushin
from mushin import _slurm

# One record per line, as `scontrol --all --oneliner show partition` prints
# it. The first is the "show part debug" example of SLURM's scontrol man page
# (https://slurm.schedmd.com/scontrol.html), its lines joined; the others
# vary its MaxTime over the documented output format, "[days-]hours:minutes:
# seconds" or "UNLIMITED", and one is hidden (listed only under --all).
DEBUG = (
    "PartitionName=debug AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL "
    "AllocNodes=ALL Default=YES QoS=N/A DefaultTime=NONE DisableRootJobs=NO "
    "Exclusive=NO GraceTime=0 Hidden=NO MaxNodes=UNLIMITED MaxTime=UNLIMITED "
    "MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED Nodes=snowflake[0-48] "
    "PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO "
    "OverSubscribe=NO OverTimeLimit=NONE PreemptMode=OFF State=UP "
    "TotalCPUs=588 TotalNodes=49 SelectTypeParameters=NONE JobDefaults=(null) "
    "DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED "
    "TRES=cpu=588,mem=1204224M,node=49,billing=49"
)


def _like_debug(name: str, max_time: str, default: str = "NO", hidden: str = "NO"):
    return (
        DEBUG.replace("PartitionName=debug", f"PartitionName={name}")
        .replace("MaxTime=UNLIMITED", f"MaxTime={max_time}")
        .replace("Default=YES", f"Default={default}")
        .replace("Hidden=NO", f"Hidden={hidden}")
    )


CLUSTER = "\n".join(
    [
        _like_debug("short", "00:30:00", default="YES"),
        DEBUG.replace("Default=YES", "Default=NO"),
        _like_debug("gpu", "2-00:00:00"),
        _like_debug("night", "12:00:00", hidden="YES"),
    ]
)


@pytest.fixture(autouse=True)
def _no_cluster(monkeypatch):
    """No test asks a real cluster: the check sees no `scontrol` unless a
    test hands it partitions; and no SBATCH_* of this machine leaks in."""
    monkeypatch.setattr(_slurm, "_partitions", lambda: None)
    for name in (*_slurm._ENV.values(), "SBATCH_CLUSTERS", "SUBMITIT_EXECUTOR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cluster(monkeypatch):
    calls = []

    def partitions():
        calls.append(1)
        return _slurm._parse_partitions(CLUSTER)

    monkeypatch.setattr(_slurm, "_partitions", partitions)
    return calls


def test_a_cell_is_one_task_and_leaves_the_rest_to_slurm():
    assert mushin.slurm() == {"nodes": 1, "tasks_per_node": 1, "timeout_min": 60}


def test_the_fields_given_pass_to_the_launcher():
    config = mushin.slurm(
        gpus=2,
        cpus=8,
        mem_gb=32,
        timeout_min=240,
        partition="gpu",
        account="lab",
        qos="normal",
        setup=["module load cuda/12", "source $HOME/.venv/bin/activate"],
        array_parallelism=16,
        constraint="a100",
    )
    assert config == {
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task": 8,
        "timeout_min": 240,
        "gpus_per_node": 2,
        "mem_gb": 32,
        "partition": "gpu",
        "account": "lab",
        "qos": "normal",
        "setup": ["module load cuda/12", "source $HOME/.venv/bin/activate"],
        "array_parallelism": 16,
        "constraint": "a100",
    }


@pytest.mark.parametrize(
    ("given", "extra"),
    [
        ({}, dict(nodes=2)),
        ({}, dict(tasks_per_node=2)),
        (dict(cpus=4), dict(cpus_per_task=8)),
        (dict(gpus=1), dict(gpus_per_node=4)),
    ],
    ids=str,
)
def test_a_field_the_helper_wrote_is_not_taken_again(given, extra):
    # In `extra` it would override the job of one task, or an argument given,
    # unchecked.
    with pytest.raises(ValueError, match="already writes"):
        mushin.slurm(**given, **extra)


def test_other_fields_pass_through_unchecked():
    # A typed GPU request, or memory per CPU, when gpus= and mem_gb= are unset.
    config = mushin.slurm(
        gres="gpu:a100:1",
        mem_per_cpu=4,
        additional_parameters={"gres-flags": "enforce-binding"},
    )
    assert config["gres"] == "gpu:a100:1" and config["mem_per_cpu"] == 4
    assert config["additional_parameters"] == {"gres-flags": "enforce-binding"}


@pytest.mark.parametrize(
    "bad",
    [
        dict(gpus=-1),
        dict(cpus=0),
        dict(timeout_min=0),
        dict(mem_gb=0.5),  # `--mem=0` would ask for a whole node's
        dict(mem_gb=1.5),
        dict(cpus=True),
        dict(timeout_min="60"),
        dict(partition=""),
        dict(account=3),
        dict(timeout_min=math.inf),
    ],
    ids=str,
)
def test_a_request_out_of_shape_is_refused(bad):
    with pytest.raises(ValueError, match="must be"):
        mushin.slurm(**bad)


def test_a_whole_float_or_a_numpy_integer_is_taken_as_its_int():
    np = pytest.importorskip("numpy")
    config = mushin.slurm(mem_gb=4.0, cpus=np.int64(2), gpus=np.int32(1))
    assert (config["mem_gb"], config["cpus_per_task"], config["gpus_per_node"]) == (
        4,
        2,
        1,
    )
    assert type(config["cpus_per_task"]) is int


@pytest.mark.parametrize(
    "setup",
    [
        "module load cuda",
        ["module load cuda", 3],
        (f"module load {m}" for m in ["cuda"]),
        {"module load cuda": 1},
    ],
    ids=["a string", "not all strings", "a generator", "a dict"],
)
def test_setup_must_be_a_list_of_lines(setup):
    with pytest.raises(ValueError, match="list of shell lines"):
        mushin.slurm(setup=setup)


@pytest.mark.parametrize(
    "line",
    [
        "export DATA=${HOME}/data",
        "cd ${SLURM_SUBMIT_DIR:-/tmp}",
        "echo ${#PATH}",
        "cd ${hydra:runtime.cwd}",  # Hydra's own: changed by the writer too
    ],
)
def test_a_braced_variable_in_setup_is_refused(line):
    # run() writes launcher_config as Hydra overrides; OmegaConf reads any
    # `${...}` as its own interpolation, and no escape survives the writer.
    with pytest.raises(ValueError, match="own interpolation"):
        mushin.slurm(setup=[line])


def test_every_key_is_a_field_of_the_plugins_slurm_config():
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
    from omegaconf import OmegaConf

    fields = set(OmegaConf.structured(SlurmQueueConf).keys())
    config = mushin.slurm(
        gpus=1, mem_gb=8, partition="p", account="a", qos="q", setup=["true"]
    )
    assert set(config) <= fields


@pytest.mark.skipif(sys.platform == "win32", reason="a stand-in srun script")
def test_the_job_script_runs_one_task_and_the_setup_lines_before_srun(
    tmp_path, monkeypatch
):
    """The plugin's translation of the config, rendered by submitit as the
    sbatch script it would submit (submitit wants an `srun` on PATH)."""
    submitit = pytest.importorskip("submitit")
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
    from omegaconf import OmegaConf

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "srun").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    setup = ["module load cuda/12", "echo $HOME 'a b'"]
    config = mushin.slurm(gpus=2, cpus=4, timeout_min=90, partition="gpu", setup=setup)
    base = set(OmegaConf.structured(BaseQueueConf).keys())
    params = {k if k in base else f"slurm_{k}": v for k, v in config.items()}
    executor = submitit.AutoExecutor(folder=tmp_path / "jobs", cluster="slurm")
    executor.update_parameters(**params)
    script = executor._executor._make_submission_file_text("python -m task", "uid")
    lines = script.splitlines()
    for flag in (
        "--nodes=1",
        "--ntasks-per-node=1",
        "--gpus-per-node=2",
        "--cpus-per-task=4",
        "--time=90",
        "--partition=gpu",
    ):
        assert f"#SBATCH {flag}" in lines
    srun = next(i for i, line in enumerate(lines) if line.startswith("srun"))
    where = [lines.index(line) for line in setup]  # verbatim, in order, before srun
    assert where == sorted(where) and where[-1] < srun


def test_the_setup_lines_reach_the_launcher_as_written():
    """Through the override text run() writes, as Hydra composes it."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra import compose, initialize

    from mushin.workflows import _to_override_element

    setup = [
        "export X=$HOME",
        "echo 'a b' \"c\"",
        r"printf 'a\n'",
        "source ~/.venv/bin/activate",
    ]
    override = "hydra.launcher.setup=" + _to_override_element(
        mushin.slurm(setup=setup)["setup"]
    )
    with initialize(version_base="1.3", config_path=None):
        cfg = compose(
            overrides=["hydra/launcher=submitit_slurm", override],
            return_hydra_config=True,
        )
    assert list(cfg.hydra.launcher.setup) == setup


# The environment sbatch reads over the job script.


@pytest.mark.parametrize(
    ("given", "name", "value"),
    [
        (dict(partition="here"), "SBATCH_PARTITION", "elsewhere"),
        (dict(qos="here"), "SBATCH_QOS", "elsewhere"),
        (dict(account="here"), "SBATCH_ACCOUNT", "elsewhere"),
        (dict(timeout_min=600), "SBATCH_TIMELIMIT", "60"),
    ],
)
def test_an_environment_that_overrides_an_argument_warns(
    monkeypatch, given, name, value
):
    monkeypatch.setenv(name, value)
    with pytest.warns(UserWarning, match=f"{name}='{value}'.*overrides"):
        mushin.slurm(**given)


@pytest.mark.parametrize(
    "given", [dict(timeout_min=60), {}], ids=["the same minutes", "not given"]
)
def test_a_time_limit_the_environment_agrees_with_does_not_warn(
    monkeypatch, recwarn, given
):
    monkeypatch.setenv("SBATCH_TIMELIMIT", "1:00:00")
    mushin.slurm(**given)
    assert not [w for w in recwarn if "overrides" in str(w.message)]


def test_inside_a_job_nothing_warns(monkeypatch, recwarn):
    monkeypatch.setenv("SUBMITIT_EXECUTOR", "slurm")
    monkeypatch.setenv("SBATCH_PARTITION", "elsewhere")
    mushin.slurm(partition="here")
    assert not [w for w in recwarn if "overrides" in str(w.message)]


# The check against `scontrol`.


def test_an_unknown_partition_is_refused_with_the_ones_that_fit(cluster):
    with pytest.raises(ValueError) as info:
        mushin.slurm(partition="gpus", timeout_min=60)
    message = str(info.value)
    assert "no partition 'gpus'" in message
    assert "debug, gpu, night" in message  # 'short' allows 30 minutes
    assert "check=False" in message


def test_a_hidden_partition_counts(cluster):
    mushin.slurm(partition="night", timeout_min=600)


@pytest.mark.parametrize(
    ("partition", "minutes", "refused"),
    [
        ("short", 30, False),
        ("short", 31, True),
        ("gpu", 2880, False),
        ("gpu", 2881, True),
        ("debug", 10**6, False),  # UNLIMITED
    ],
)
def test_a_time_past_the_partitions_limit_is_refused(
    cluster, partition, minutes, refused
):
    if refused:
        with pytest.raises(ValueError, match="past partition") as info:
            mushin.slurm(partition=partition, timeout_min=minutes)
        assert "check=False" in str(info.value)
    else:
        mushin.slurm(partition=partition, timeout_min=minutes)


def test_the_environment_sbatch_will_read_is_what_is_judged(cluster, monkeypatch):
    # SBATCH_PARTITION decides the partition, over partition=.
    monkeypatch.setenv("SBATCH_PARTITION", "short")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="'short'"):
        mushin.slurm(partition="gpu", timeout_min=60)
    monkeypatch.setenv("SBATCH_PARTITION", "gpux")
    with pytest.raises(ValueError, match="'gpux' \\(from SBATCH_PARTITION\\)"):
        mushin.slurm(timeout_min=60)
    monkeypatch.delenv("SBATCH_PARTITION")
    # SBATCH_TIMELIMIT decides the time limit, over timeout_min=.
    monkeypatch.setenv("SBATCH_TIMELIMIT", "1:00:00")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="60 minutes"):
        mushin.slurm(partition="short", timeout_min=10)
    # SBATCH_TIMELIMIT=0 asks sbatch for no time limit.
    monkeypatch.setenv("SBATCH_TIMELIMIT", "0")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="past partition"):
        mushin.slurm(partition="short", timeout_min=10)
    monkeypatch.setenv("SBATCH_TIMELIMIT", "1:00:00")
    # SBATCH_QOS is a QoS, which may lift the partition's limit.
    monkeypatch.setenv("SBATCH_QOS", "long")
    with pytest.warns(UserWarning, match="SBATCH_TIMELIMIT"):
        mushin.slurm(partition="short", timeout_min=10)


def test_what_scontrol_does_not_state_is_not_judged(cluster, monkeypatch):
    # No partition: the default ('short', 30 minutes) is not assumed, since
    # the cluster may route a job that names none.
    mushin.slurm(timeout_min=10**5)
    mushin.slurm(partition="short", timeout_min=600, qos="long")  # QoS may lift it
    mushin.slurm(partition="short,gpu", timeout_min=600)  # SLURM picks among them
    mushin.slurm(partition="short", timeout_min=600, check=False)
    for option in ("partition", "time", "qos", "clusters"):
        # additional_parameters, written last, decide instead.
        mushin.slurm(
            partition="nowhere", additional_parameters={option: "x"}, timeout_min=60
        )
    # SBATCH_CLUSTERS sends the jobs to a cluster scontrol is not asked about.
    monkeypatch.setenv("SBATCH_CLUSTERS", "other")
    mushin.slurm(partition="nowhere", timeout_min=60)
    monkeypatch.delenv("SBATCH_CLUSTERS")
    # Inside a job (an import of a module that builds its config), never.
    monkeypatch.setenv("SUBMITIT_EXECUTOR", "slurm")
    calls = len(cluster)
    mushin.slurm(partition="nowhere")
    assert len(cluster) == calls


def test_an_unparsable_time_limit_is_not_judged(monkeypatch):
    monkeypatch.setattr(
        _slurm,
        "_partitions",
        lambda: _slurm._parse_partitions(_like_debug("x", "soon")),
    )
    mushin.slurm(partition="x", timeout_min=10**6)


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("UNLIMITED", math.inf),
        ("INFINITE", math.inf),
        ("30", 30),  # minutes
        ("30:30", 30.5),  # minutes:seconds
        ("2:00:00", 120),  # hours:minutes:seconds
        ("00:30:00", 30),
        ("1-12", 2160),  # days-hours
        ("1-12:30", 2190),  # days-hours:minutes
        ("2-00:00:00", 2880),  # days-hours:minutes:seconds
        ("NONE", None),
        ("", None),
    ],
)
def test_slurms_time_formats(text, minutes):
    assert _slurm._minutes(text) == minutes


@pytest.fixture
def real_partitions(monkeypatch):
    """The real `_partitions`, its runner stood in, and its kept answer
    cleared before and after (even when the test fails)."""
    monkeypatch.setattr(_slurm, "_partitions", REAL_PARTITIONS)
    monkeypatch.setattr(_slurm, "_ANSWER", None)
    yield


REAL_PARTITIONS = _slurm._partitions


def test_scontrol_missing_slow_failing_or_garbled_judges_nothing(
    monkeypatch, real_partitions
):
    monkeypatch.setattr(_slurm, "_ANSWER_TTL_S", 0)  # ask anew each time
    monkeypatch.setattr(_slurm.shutil, "which", lambda name: None)
    assert _slurm._partitions() is None

    monkeypatch.setattr(_slurm.shutil, "which", lambda name: "/usr/bin/scontrol")

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(_slurm.subprocess, "run", slow)
    assert _slurm._partitions() is None

    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 1, "", "Unable to contact slurm controller"
        )

    monkeypatch.setattr(_slurm.subprocess, "run", failing)
    assert _slurm._partitions() is None

    seen = []

    def answering(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, CLUSTER, "")

    monkeypatch.setattr(_slurm.subprocess, "run", answering)
    names = [p["PartitionName"] for p in _slurm._partitions()]
    assert names == ["short", "debug", "gpu", "night"]
    cmd, kwargs = seen[0]
    assert cmd[1:] == ["--all", "--oneliner", "show", "partition"]
    assert kwargs["timeout"] == _slurm._SCONTROL_TIMEOUT_S
    assert kwargs["errors"] == "replace"  # a byte not UTF-8 is not an error
    assert "shell" not in kwargs


def test_an_answer_is_kept_for_a_while(monkeypatch, real_partitions):
    monkeypatch.setattr(_slurm.shutil, "which", lambda name: "/usr/bin/scontrol")
    seen = []

    def failing(*args, **kwargs):
        seen.append(1)
        return subprocess.CompletedProcess(args[0], 1, "", "")

    monkeypatch.setattr(_slurm.subprocess, "run", failing)
    clock = iter([0.0, 1.0, _slurm._ANSWER_TTL_S + 1.0])
    monkeypatch.setattr(_slurm.time, "monotonic", lambda: next(clock))
    _slurm._partitions()
    _slurm._partitions()  # within the TTL: not asked again
    assert len(seen) == 1
    _slurm._partitions()  # past it: asked again
    assert len(seen) == 2
