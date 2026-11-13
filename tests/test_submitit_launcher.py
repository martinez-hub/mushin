# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""mushin's submitit launchers as a Hydra plugin (#345).

The sweeps that run on them are in tests/test_submitit_local.py; these tests
cover the plugin module itself, which Hydra imports in every application, and
how `run()` chooses it.
"""

import hashlib
import inspect
import subprocess
import sys
import textwrap

import pytest

from mushin.workflows import _route_to_mushin_launchers


def test_without_the_plugin_the_module_imports_silently_and_defines_nothing():
    """Hydra imports every `hydra_plugins` module whenever any Hydra app
    starts; an ImportError there is a warning in every one of them."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["hydra_plugins.hydra_submitit_launcher"] = None  # not installed
        import hydra_plugins.mushin_submitit.launcher as m
        assert not hasattr(m, "MushinSlurmLauncher"), dir(m)
        from mushin.workflows import _route_to_mushin_launchers
        o = ["hydra/launcher=submitit_slurm"]
        assert _route_to_mushin_launchers(o) == o  # Hydra names what was typed
        """
    )
    done = subprocess.run(
        [sys.executable, "-W", "error", "-c", code], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_hydra_finds_both_launchers_and_scans_without_a_warning(recwarn):
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra.core.plugins import Plugins
    from hydra.plugins.launcher import Launcher

    found = {c.__name__ for c in Plugins.instance().discover(Launcher)}
    assert {"MushinSlurmLauncher", "MushinLocalLauncher"} <= found
    assert not [w for w in recwarn if "mushin_submitit" in str(w.message)]


def test_mushins_config_has_the_plugins_fields_and_only_another_target():
    """`launcher_config` and `submitit_slurm_config` keys stay valid."""
    import dataclasses

    stock = pytest.importorskip("hydra_plugins.hydra_submitit_launcher.config")
    from hydra_plugins.mushin_submitit import launcher as ours

    for mine, theirs in [
        (ours.MushinSlurmQueueConf, stock.SlurmQueueConf),
        (ours.MushinLocalQueueConf, stock.LocalQueueConf),
    ]:
        a = {f.name: f.default for f in dataclasses.fields(mine)}
        b = {f.name: f.default for f in dataclasses.fields(theirs)}
        assert a.keys() == b.keys()
        assert {k for k in a if a[k] != b[k]} == {"_target_"}
        assert a["_target_"].startswith("hydra_plugins.mushin_submitit.launcher.")


def test_the_submission_is_the_plugins_own():
    """mushin copies the plugin's `launch()` up to its submission. A plugin
    release that changes `launch()` fails the digest: compare the new body
    with `_submit`, update the copy, then the digest."""
    stock = pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
        BaseSubmititLauncher,
    )

    from hydra_plugins.mushin_submitit import launcher as ours

    theirs = inspect.getsource(BaseSubmititLauncher.launch)
    # Code lines only: a reformat or a comment does not change what runs.
    code = "\n".join(
        line.strip()
        for line in theirs.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert hashlib.sha256(code.encode()).hexdigest() == (
        "51b35854f952463663508f230f5a860c435689c329bb2a184b1cc876ffb69a2e"
    ), f"hydra-submitit-launcher {stock.__version__} changed launch()"

    def body(source, start, end):
        lines = textwrap.dedent(source).splitlines()
        i = next(n for n, line in enumerate(lines) if start in line)
        j = next(n for n, line in enumerate(lines) if end in line)
        # Lint comments on the copy are not part of the code.
        return "\n".join(line.split("  # noqa")[0].strip() for line in lines[i : j + 1])

    copied = body(
        inspect.getsource(ours._MushinSubmititMixin._submit),
        "import submitit",
        "jobs = executor.map_array",
    )
    original = body(theirs, "import submitit", "jobs = executor.map_array")
    # The copy's only edits: names it reaches through the plugin's modules,
    # and the builtin generic.
    original = original.replace(
        "structured(BaseQueueConf)", "structured(_plugin_config.BaseQueueConf)"
    ).replace("List[Any]", "list[Any]")
    assert copied == original


def test_the_job_is_composed_as_hydras_run_job_composes_it():
    """`_dead_return` mirrors the part of Hydra's `run_job` that places a job
    (its dir, `runtime.output_dir`, the JobReturn's fields, the chdir rule).
    A hydra-core release that changes that part fails here: compare it with
    `_dead_return`, then update the digest."""
    from hydra.core.utils import run_job

    lines = inspect.getsource(run_job).splitlines()
    i = next(n for n, line in enumerate(lines) if "job_dir_key))" in line)
    j = next(
        n for n, line in enumerate(lines) if "ret.working_dir = os.getcwd()" in line
    )
    code = "\n".join(
        line.strip()
        for line in lines[i : j + 1]
        if line.strip() and not line.strip().startswith("#")
    )
    assert hashlib.sha256(code.encode()).hexdigest() == (
        "7990aa89eac439de7d4fffa47c6f5efa2c54eceb00d79664060797ee743e91fa"
    )


def test_recording_a_dead_cell_never_fails_the_run(tmp_path):
    """Recording is best-effort: whatever goes wrong there, the run still
    has the failure through the job's return."""
    from hydra.core.utils import JobReturn
    from omegaconf import OmegaConf

    from mushin.workflows import _DeadJob, _record_dead_cell

    ret = JobReturn()
    ret.working_dir = "0"
    # A hydra node with no `output_subdir` or `job`: reading them raises.
    ret.hydra_cfg = OmegaConf.create({"hydra": {"runtime": {"cwd": str(tmp_path)}}})
    OmegaConf.set_struct(ret.hydra_cfg, True)
    dead = _DeadJob(RuntimeError("x"), "RuntimeError('x')")
    _record_dead_cell(
        ret,
        dead,
        combo={},
        status="failed",
        on_error="nan",
        prior={},
        run_id="r",
    )


def test_a_dead_cell_records_this_sweeps_combo(tmp_path):
    """A `running` record an earlier sweep left in a reused dir names that
    sweep's combo; the failed record names this one's."""
    from hydra.core.utils import JobReturn
    from omegaconf import OmegaConf

    from mushin._resume import read_cell_status, write_cell_status
    from mushin.workflows import _DeadJob, _record_dead_cell

    write_cell_status(tmp_path / "0", status="running", combo={"lr": 0.1}, attempt=1)
    ret = JobReturn()
    ret.working_dir = "0"
    ret.hydra_cfg = OmegaConf.create(
        {
            "hydra": {
                "runtime": {"cwd": str(tmp_path)},
                "output_subdir": None,
                "job": {"num": 0},
            }
        }
    )
    dead = _DeadJob(RuntimeError("x"), "RuntimeError('x')")
    # This sweep's combo, even an empty one (nothing swept); the prior
    # record's only when this one could not be computed.
    for combo, recorded in (({"lr": 0.2}, {"lr": 0.2}), ({}, {}), (None, {"lr": 0.1})):
        write_cell_status(
            tmp_path / "0", status="running", combo={"lr": 0.1}, attempt=1
        )
        _record_dead_cell(
            ret,
            dead,
            combo=combo,
            status="failed",
            on_error="raise",
            prior=read_cell_status(tmp_path / "0"),
            run_id="r",
        )
        status = read_cell_status(tmp_path / "0")
        assert status["status"] == "failed" and status["combo"] == recorded


def _dead_ret(root, *, output_subdir=None, job_id="7", stamp="this"):
    from hydra.core.utils import JobReturn
    from omegaconf import OmegaConf

    ret = JobReturn()
    ret.working_dir = "0"
    ret.cfg = OmegaConf.create({"lr": 0.2})
    ret.overrides = ["+lr=0.2"]
    ret.hydra_cfg = OmegaConf.create(
        {
            "hydra": {
                "runtime": {"cwd": str(root)},
                "output_subdir": output_subdir,
                "job": {"num": 0, "id": job_id, "mushin_run_id": stamp},
            }
        }
    )
    return ret


def test_a_dead_cell_is_recorded_as_350_did_with_its_run_and_job(tmp_path):
    """Over a 'running' record the attempt, fingerprints and snapshot stay,
    as #350 recorded them; the record now names this run and the job."""
    from mushin._resume import read_cell_status, write_cell_status
    from mushin.workflows import _DeadJob, _record_dead_cell

    cell = tmp_path / "0"
    snapshot = cell / ".hydra"
    snapshot.mkdir(parents=True)
    # The job's own: Hydra wrote it, naming this run, before the runner's
    # 'running' record.
    (snapshot / "hydra.yaml").write_text("hydra:\n  job:\n    mushin_run_id: this\n")
    (snapshot / "config.yaml").write_text("lr: 0.1\n")
    write_cell_status(
        cell,
        status="running",
        combo={"lr": 0.2},
        attempt=3,
        config_hash="c",
        code_hash="k",
        run_id="this",
    )
    _record_dead_cell(
        _dead_ret(tmp_path, output_subdir=".hydra"),
        _DeadJob(RuntimeError("x"), "RuntimeError('died')"),
        combo={"lr": 0.2},
        status="failed",
        on_error="raise",
        prior=read_cell_status(cell),
        run_id="this",
    )
    status = read_cell_status(cell)
    assert status["status"] == "failed" and status["attempt"] == 3
    assert (status["config_hash"], status["code_hash"]) == ("c", "k")
    assert status["run_id"] == "this" and status["job_id"] == "7"
    assert (snapshot / "config.yaml").read_text() == "lr: 0.1\n"


@pytest.mark.parametrize(
    "earlier",
    ["hydra:\n  job:\n    mushin_run_id: earlier\n", "earlier: true\n"],
    ids=["another-run", "before-the-stamp"],
)
def test_a_dead_cell_gets_this_jobs_snapshot_over_an_earlier_runs(tmp_path, earlier):
    """A job that died before Hydra made its dir, in a dir an earlier run
    used: the snapshot there names another run (or none, from before the
    stamp), so it is this job's that a reload must find beside its record."""
    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin._resume import read_cell_status, write_cell_status
    from mushin.workflows import _DeadJob, _record_dead_cell

    cell = tmp_path / "0"
    snapshot = cell / ".hydra"
    snapshot.mkdir(parents=True)
    (snapshot / "hydra.yaml").write_text(earlier)
    (snapshot / "config.yaml").write_text("lr: 0.1\n")
    write_cell_status(cell, status="running", combo={"lr": 0.1}, attempt=1)
    _record_dead_cell(
        _dead_ret(tmp_path, output_subdir=".hydra"),
        _DeadJob(RuntimeError("x"), "RuntimeError('died')"),
        combo={"lr": 0.2},
        status="failed",
        on_error="raise",
        prior=read_cell_status(cell),
        run_id="this",
    )
    assert hydra_run_stamp(load_hydra_record(snapshot / "hydra.yaml")) == "this"
    assert "lr: 0.2" in (snapshot / "config.yaml").read_text()
    assert read_cell_status(cell)["run_id"] == "this"


@pytest.mark.parametrize(
    ("given", "routed"),
    [
        (["hydra/launcher=submitit_slurm"], ["hydra/launcher=mushin_submitit_slurm"]),
        (["hydra/launcher=submitit_local"], ["hydra/launcher=mushin_submitit_local"]),
        (["+hydra/launcher=submitit_slurm"], ["+hydra/launcher=mushin_submitit_slurm"]),
        (["hydra/launcher= submitit_slurm"], ["hydra/launcher=mushin_submitit_slurm"]),
        (["hydra/launcher=joblib"], ["hydra/launcher=joblib"]),
        (["hydra/launcher=basic"], ["hydra/launcher=basic"]),
        (["x=submitit_slurm"], ["x=submitit_slurm"]),
        (["hydra.launcher.timeout_min=5"], ["hydra.launcher.timeout_min=5"]),
    ],
)
def test_only_the_submitit_launcher_choices_are_routed(given, routed):
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    assert _route_to_mushin_launchers(given) == routed


class _Job:
    """A stand-in for a submitit job: `results()` returns or raises."""

    def __init__(self, outcome, done=True):
        self.outcome, self.waited, self._done = outcome, False, done

    def done(self):
        return self._done

    def results(self):
        self.waited = True
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return [self.outcome]


@pytest.fixture
def launcher(monkeypatch):
    """mushin's local launcher for a mushin runner, with `_dead_return`
    reduced to a marker."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from hydra_plugins.mushin_submitit.launcher import MushinLocalLauncher
    from mushin.workflows import _TaskRunner

    launcher = MushinLocalLauncher()
    launcher.task_function = object.__new__(_TaskRunner)
    monkeypatch.setattr(
        launcher, "_dead_return", lambda job, overrides, idx, exc: ("dead", idx, exc)
    )
    return launcher


PARAMS = [([f"seed={i}"], "hydra.sweep.dir", 10 + i) for i in range(4)]


def test_every_job_is_waited_on_and_each_dead_one_comes_back(launcher):
    """The plugin waited in submission order and stopped at the first job
    that ended without a result; every job is waited on here, in order."""
    from submitit.core.utils import FailedJobError, UncompletedJobError

    lost, broken = UncompletedJobError("no result"), FailedJobError("unpickle")
    unreadable = EOFError("truncated result")
    jobs = [_Job(lost), _Job("r1"), _Job(broken), _Job(unreadable, done=True)]
    returns = launcher._wait(jobs, PARAMS)
    assert all(j.waited for j in jobs)
    assert returns == [
        ("dead", 10, lost),
        "r1",
        ("dead", 12, broken),
        ("dead", 13, unreadable),
    ]

    # Any other task function gets the plugin's behavior: the error, raised.
    launcher.task_function = print
    with pytest.raises(UncompletedJobError):
        launcher._wait([_Job(lost), _Job("r1")], PARAMS[:2])


def test_an_error_it_cannot_record_is_raised_after_every_job(launcher, monkeypatch):
    """A job not known to have ended is not recorded (its cell may still be
    written to), nor one whose cell does not compose; each is raised once
    every job has been waited on."""
    from submitit.core.utils import UncompletedJobError

    unsure = OSError("scheduler did not answer")
    jobs = [_Job(unsure, done=False), _Job("r1"), _Job(UncompletedJobError("x"))]
    with pytest.raises(OSError, match="scheduler"):
        launcher._wait(jobs, PARAMS[:3])
    assert all(j.waited for j in jobs)

    def no_compose(job, overrides, idx, exc):
        raise KeyError("cannot compose")

    monkeypatch.setattr(launcher, "_dead_return", no_compose)
    jobs = [_Job(UncompletedJobError("x")), _Job("r1")]
    with pytest.raises(KeyError, match="compose"):
        launcher._wait(jobs, PARAMS[:2])
    assert all(j.waited for j in jobs)


def test_a_long_submitit_message_keeps_its_start_and_end():
    """submitit puts a dead job's whole error stream in its error, which a
    long run can make megabytes long; mushin keeps a bounded copy."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from submitit.core.utils import FailedJobError, UncompletedJobError

    from hydra_plugins.mushin_submitit.launcher import _bounded

    short, other = UncompletedJobError("no result"), EOFError("x" * 100_000)
    assert _bounded(short, "log") is short and _bounded(other, "log") is other
    for kind in (UncompletedJobError, FailedJobError):
        try:
            raise kind("head\n" + "x" * 1_000_000 + "\nkilled: out of memory")
        except kind as raised:
            long = raised
        kept = _bounded(long, "/logs")
        assert type(kept) is kind and len(str(kept)) < 23_000
        assert str(kept).startswith("head\n")
        assert str(kept).endswith("killed: out of memory")
        # The cut text may be stderr or stdout: name the folder of both.
        assert "cut by mushin; the job's logs are in /logs" in str(kept)
        # No traceback: the original's frames hold the whole message.
        assert long.__traceback__ is not None and kept.__traceback__ is None


class _Paths:
    folder = "/logs"
    stderr, stdout = "/logs/1_0_log.err", "/logs/1_0_log.out"


class _Ended:
    """A submitit job that has ended, for `_summary`."""

    job_id, state, paths = "1", "COMPLETED", _Paths()

    def __init__(self, num_tasks=1):
        self.num_tasks = num_tasks


def _failed_job_error(trace):
    """A FailedJobError as submitit's `Job.exception()` builds it."""
    from submitit.core.utils import FailedJobError

    rule = "----------------------"
    return FailedJobError(
        f"Job (task=0) failed during processing with trace:\n{rule}\n{trace}\n"
        f"{rule}\nYou can check full logs with 'job.stderr(0)'"
    )


def test_the_summary_names_what_ended_the_job_and_its_log():
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from submitit.core.utils import UncompletedJobError

    from hydra_plugins.mushin_submitit.launcher import _summary

    lost = _summary(_Ended(), UncompletedJobError("no output\nlog"), {})
    assert lost == (
        "UncompletedJobError('submitit job 1 gave no result (state: COMPLETED);"
        " its log: /logs/1_0_log.err')"
    )
    assert "its log: /logs/1_0_log.out" in _summary(
        _Ended(), UncompletedJobError("x"), {"stderr_to_stdout": True}
    )

    # A FailedJobError: the job pickled an error instead of a result. The
    # line does not name it (the traceback is in mushin_error.txt).
    trace = "Traceback (most recent call last):\n  File ...\nModuleNotFoundError: x\n"
    raised = _summary(_Ended(), _failed_job_error(trace), {})
    assert raised == (
        "FailedJobError('submitit job 1 raised an error instead of returning"
        " (state: COMPLETED); its log: /logs/1_0_log.err')"
    )

    # A job of several tasks has a log per task: name their folder.
    several = _summary(_Ended(num_tasks=2), UncompletedJobError("x"), {})
    assert "its log: /logs (one per task)" in several


THIS, OTHER = {"lr": 0.2}, {"lr": 0.1}


@pytest.mark.parametrize(
    ("prior", "metrics", "keepable", "returns", "recorded"),
    [
        # This run's record of this cell.
        (
            dict(status="completed", run_id="this", metrics="OURS"),
            True,
            True,
            "metrics",
            False,
        ),
        (
            dict(status="completed", run_id="this", metrics="OURS"),
            True,
            False,
            "dead",
            False,
        ),
        (dict(status="completed", run_id="this"), False, True, "dead", False),
        # A file the record does not claim (the result was not whole).
        (dict(status="completed", run_id="this"), True, True, "dead", False),
        # A file other than the one it wrote (another job's, in a shared dir).
        (
            dict(status="completed", run_id="this", metrics="0" * 16),
            True,
            True,
            "dead",
            False,
        ),
        (
            dict(status="completed", run_id="this", metrics="OURS"),
            False,
            True,
            "dead",
            False,
        ),
        (dict(status="skipped", run_id="this"), False, True, "skipped", False),
        (dict(status="failed", run_id="this"), False, True, "dead", False),
        (dict(status="running", run_id="this"), False, True, "dead", True),
        # This run's record of another cell in the same dir.
        (
            dict(status="completed", run_id="this", combo=OTHER, metrics="OURS"),
            True,
            True,
            "dead",
            False,
        ),
        # An earlier run's records.
        (
            dict(status="completed", run_id="earlier", metrics="OURS"),
            True,
            True,
            "dead",
            False,
        ),
        (
            dict(status="completed", run_id="earlier", combo=OTHER),
            True,
            True,
            "dead",
            False,
        ),
        (dict(status="failed", run_id="earlier"), False, True, "dead", False),
        (
            dict(status="failed", run_id="earlier", combo=OTHER),
            False,
            True,
            "dead",
            False,
        ),
        (
            dict(status="running", run_id="earlier", combo=OTHER),
            False,
            True,
            "dead",
            True,
        ),
        (None, False, True, "dead", True),
    ],
)
def test_a_dead_job_keeps_its_own_end_and_records_only_over_what_it_may(
    tmp_path, prior, metrics, keepable, returns, recorded
):
    """Which record a dead job keeps as its own end, and which it records its
    end over or leaves, as #350 did (another cell's records: #355)."""
    from hydra.core.utils import JobStatus

    from mushin._resume import read_cell_status, write_cell_status
    from mushin.workflows import _DeadJob, _SkippedRun, _TaskRunner

    cell = tmp_path / "0"
    cell.mkdir()
    from mushin._sweep_io import _digest

    file = '{"v": 1.0}'
    if prior is not None:
        fields = {"combo": THIS, "attempt": 1, **prior}
        if fields.get("metrics") == "OURS":
            fields["metrics"] = _digest(file)
        write_cell_status(cell, **fields)
        (cell / "mushin_error.txt").write_text("boom\n")
    if metrics:
        (cell / "mushin_metrics.json").write_text(file)
    before = read_cell_status(cell)

    runner = object.__new__(_TaskRunner)
    runner.swept_names, runner.sampled_indices = ["lr"], None
    runner.on_error, runner.run_id = "nan", "this"
    dead = _DeadJob(RuntimeError("x"), "RuntimeError('died')", keepable=keepable)
    status, value = runner.record_dead_cell(_dead_ret(tmp_path), dead)

    assert status is JobStatus.COMPLETED
    if returns == "metrics":
        assert value == {"v": 1.0}
    elif returns == "skipped":
        assert isinstance(value, _SkippedRun)
    else:
        assert value is dead
    after = read_cell_status(cell)
    if recorded:
        assert after["status"] == "failed" and after["combo"] == THIS
        assert after["run_id"] == "this" and after["job_id"] == "7"
        assert "died" in (cell / "mushin_error.txt").read_text()
        assert not (cell / "mushin_metrics.json").exists()
    else:
        assert after == before
        if prior is not None:
            assert (cell / "mushin_error.txt").read_text() == "boom\n"
        assert (cell / "mushin_metrics.json").exists() == metrics


def test_only_a_one_task_job_that_gave_no_result_may_keep_its_record():
    from submitit.core.utils import UncompletedJobError

    from hydra_plugins.mushin_submitit import launcher as ours

    gave_none = UncompletedJobError("Job 1 (task: 0) has not produced any output")
    assert ours._keepable(_Ended(), gave_none)
    assert ours._keepable(_Ended(), OSError("unreadable result"))
    assert not ours._keepable(_Ended(num_tasks=2), gave_none)
    assert not ours._keepable(_Ended(), _failed_job_error("RuntimeError: x"))
