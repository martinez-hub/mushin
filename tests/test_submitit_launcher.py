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
