# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run(..., notes=..., tags=[...])` annotates a sweep with free-form lineage:
recorded in the manifest, exposed on the workflow, and carried on the dataset."""

from __future__ import annotations

import json

import pytest

from mushin import multirun
from mushin._sweep_io import MANIFEST_FILE
from mushin.workflows import MultiRunMetricsWorkflow


class _W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(v=float(x))


def test_notes_and_tags_exposed_on_workflow(tmp_path):
    wf = _W()
    wf.run(
        x=multirun([1, 2]),
        notes="baseline before the lr fix",
        tags=["baseline", "sprint-3"],
        working_dir=str(tmp_path / "s"),
    )
    assert wf.notes == "baseline before the lr fix"
    assert wf.tags == ["baseline", "sprint-3"]


def test_notes_and_tags_persist_in_the_manifest(tmp_path):
    wd = tmp_path / "s"
    _W().run(x=multirun([1, 2]), notes="run A", tags=["exp"], working_dir=str(wd))
    manifest = json.loads((wd / MANIFEST_FILE).read_text())
    assert manifest["notes"] == "run A"
    assert manifest["tags"] == ["exp"]


def test_notes_and_tags_on_dataset_attrs(tmp_path):
    wf = _W()
    wf.run(
        x=multirun([1, 2]),
        notes="carried",
        tags=["t1", "t2"],
        working_dir=str(tmp_path / "s"),
    )
    ds = wf.to_xarray()
    assert ds.attrs["mushin_notes"] == "carried"
    assert json.loads(ds.attrs["mushin_tags"]) == ["t1", "t2"]


def test_defaults_are_empty(tmp_path):
    wf = _W()
    wf.run(x=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    assert wf.notes is None
    assert wf.tags == []
    ds = wf.to_xarray()
    assert "mushin_notes" not in ds.attrs
    assert "mushin_tags" not in ds.attrs


def test_tags_must_be_a_list_of_strings(tmp_path):
    with pytest.raises(ValueError, match="tags"):
        _W().run(x=multirun([1, 2]), tags="oops", working_dir=str(tmp_path / "s"))
    with pytest.raises(ValueError, match="tags"):
        _W().run(x=multirun([1, 2]), tags=[1, 2], working_dir=str(tmp_path / "s2"))


def test_notes_must_be_a_string(tmp_path):
    with pytest.raises(ValueError, match="notes"):
        _W().run(x=multirun([1, 2]), notes=123, working_dir=str(tmp_path / "s"))


def test_notes_and_tags_survive_reload(tmp_path):
    wd = tmp_path / "s"
    _W().run(
        x=multirun([1, 2]),
        notes="original run",
        tags=["baseline"],
        working_dir=str(wd),
    )
    loaded = MultiRunMetricsWorkflow(working_dir=wd)
    assert loaded.notes == "original run"
    assert loaded.tags == ["baseline"]


def test_notes_and_tags_survive_resume_without_repassing(tmp_path):
    # A resume that does not re-pass notes/tags must NOT wipe the lineage the
    # original run recorded.
    wd = tmp_path / "s"
    _W().run(
        x=multirun([1, 2]),
        notes="original run",
        tags=["baseline"],
        working_dir=str(wd),
    )
    wf2 = _W()
    wf2.run(x=multirun([1, 2]), resume=True, working_dir=str(wd))  # no notes/tags
    assert wf2.notes == "original run"
    assert wf2.tags == ["baseline"]
    manifest = json.loads((wd / MANIFEST_FILE).read_text())
    assert manifest["notes"] == "original run"
    assert manifest["tags"] == ["baseline"]


def test_fresh_run_does_not_inherit_prior_lineage(tmp_path):
    """The manifest fallback exists for RESUME; a fresh (non-resume) run reusing
    a working_dir is a new sweep and must not silently adopt the previous
    sweep's notes/tags."""
    wd = tmp_path / "s"
    _W().run(x=multirun([1, 2]), notes="old sweep", tags=["old"], working_dir=str(wd))
    wf2 = _W()
    wf2.run(x=multirun([1, 2]), working_dir=str(wd))  # fresh run, no lineage
    assert wf2.notes is None
    assert wf2.tags == []


def test_resume_with_empty_tags_clears_prior(tmp_path):
    """Explicitly passing `tags=[]` on a resume is a request to clear the tags,
    not an omission — it must not fall back to the prior manifest's tags."""
    wd = tmp_path / "s"
    _W().run(x=multirun([1, 2]), tags=["baseline"], working_dir=str(wd))
    wf2 = _W()
    wf2.run(x=multirun([1, 2]), resume=True, tags=[], working_dir=str(wd))
    assert wf2.tags == []
