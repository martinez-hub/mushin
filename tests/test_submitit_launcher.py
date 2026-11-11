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
    assert hashlib.sha256(theirs.encode()).hexdigest() == (
        "546875669010f80d6ceede740e8bf2d18088cf5bcce2eb4794c735fa3dc2fc0e"
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

    def __init__(self, outcome):
        self.outcome, self.waited = outcome, False

    def results(self):
        self.waited = True
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return [self.outcome]


def test_every_job_is_waited_on_and_each_dead_one_comes_back(monkeypatch):
    """The plugin waited in submission order and stopped at the first job
    that ended without a result; every job is waited on here, in order."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    from submitit.core.utils import FailedJobError, UncompletedJobError

    from hydra_plugins.mushin_submitit.launcher import MushinLocalLauncher
    from mushin.workflows import _TaskRunner

    launcher = MushinLocalLauncher()
    launcher.task_function = object.__new__(_TaskRunner)
    monkeypatch.setattr(
        launcher, "_dead_return", lambda job, overrides, idx, exc: ("dead", idx, exc)
    )
    lost, broken = UncompletedJobError("no result"), FailedJobError("unpickle")
    jobs = [_Job(lost), _Job("r1"), _Job(broken), _Job("r3")]
    params = [([f"seed={i}"], "hydra.sweep.dir", 10 + i) for i in range(4)]
    returns = launcher._wait(jobs, params)
    assert all(j.waited for j in jobs)
    assert returns == [("dead", 10, lost), "r1", ("dead", 12, broken), "r3"]

    # Any other task function gets the plugin's behavior: the error, raised.
    launcher.task_function = print
    with pytest.raises(UncompletedJobError):
        launcher._wait([_Job(lost), _Job("r1")], params[:2])
