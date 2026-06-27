import shutil
from pathlib import Path

import pytest

from mushin.study._sweep import run_training_sweep


def _make_train(tag):
    def train(seed):
        p = Path(f"_tmp_{tag}_{seed}.bin")  # saved in the Hydra job cwd
        p.write_text(f"{tag}-{seed}")
        return str(p.resolve())

    return train


@pytest.mark.usefixtures("cleandir")
def test_run_training_sweep_recovers_paths(tmp_path):
    methods = {"a": _make_train("a"), "b": _make_train("b")}
    ckpts = run_training_sweep(methods, seeds=[0, 1, 2], ckpt_dir=tmp_path / "ck")

    assert set(ckpts) == {"a", "b"}
    assert all(len(v) == 3 for v in ckpts.values())
    for m, paths in ckpts.items():
        for s, p in enumerate(paths):
            assert Path(p).exists()
            assert Path(p).read_text() == f"{m}-{s}"


@pytest.mark.usefixtures("cleandir")
def test_run_training_sweep_rejects_none_path(tmp_path):
    methods = {"a": lambda seed: None}  # train_fn that returns no path
    with pytest.raises(ValueError, match="returned no checkpoint path"):
        run_training_sweep(methods, seeds=[0], ckpt_dir=tmp_path / "ck")


@pytest.mark.usefixtures("cleandir")
def test_run_training_sweep_handles_hydra_scalar_method_names(tmp_path):
    # names that Hydra would otherwise parse as scalars or split on commas
    names = ["1", "true", "a,b"]
    methods = {n: _make_train(n.replace(",", "_")) for n in names}
    ckpts = run_training_sweep(methods, seeds=[0, 1], ckpt_dir=tmp_path / "ck")

    assert set(ckpts) == set(names)
    for n in names:
        assert len(ckpts[n]) == 2
        for p in ckpts[n]:
            assert Path(p).exists()


@pytest.mark.usefixtures("cleandir")
def test_run_training_sweep_uses_shutil_move(tmp_path, monkeypatch):
    """Confirm that shutil.move (not os.replace) is used to relocate checkpoints.

    This matters because os.replace raises OSError on cross-device moves (e.g. when
    train_fn saves to a tmpfs/NFS mount and ckpt_dir is on another filesystem).
    shutil.move falls back to copy+unlink in that case.
    """
    import mushin.study._sweep as sweep_mod

    move_calls = []
    real_move = shutil.move

    def spy_move(src, dst):
        move_calls.append((src, dst))
        return real_move(src, dst)

    monkeypatch.setattr(sweep_mod.shutil, "move", spy_move)

    methods = {"a": _make_train("a")}
    run_training_sweep(methods, seeds=[0, 1], ckpt_dir=tmp_path / "ck")

    assert len(move_calls) == 2, "shutil.move should be called once per (method, seed)"


@pytest.mark.usefixtures("cleandir")
def test_run_training_sweep_relocates_from_separate_tmp_dir(tmp_path):
    """Train fn saves to a separate temp directory; sweep must still land checkpoints.

    This exercises the relocation path when src and dest are in different
    directories (same filesystem here, but covers the shutil.move code path that
    also handles cross-device moves).
    """
    import tempfile

    def train_in_own_tmpdir(seed):
        # Each job gets its own isolated temp dir (simulates cross-dir save)
        d = Path(tempfile.mkdtemp(dir=tmp_path / "train_tmp"))
        p = d / f"ckpt_{seed}.bin"
        p.write_text(f"a-{seed}")
        return str(p)

    (tmp_path / "train_tmp").mkdir(parents=True, exist_ok=True)
    methods = {"a": train_in_own_tmpdir}
    ckpt_dir = tmp_path / "ck"
    ckpts = run_training_sweep(methods, seeds=[0, 1], ckpt_dir=ckpt_dir)

    assert set(ckpts) == {"a"}
    assert len(ckpts["a"]) == 2
    for s, p in enumerate(ckpts["a"]):
        assert Path(p).exists(), f"checkpoint for seed={s} not found at {p}"
        assert Path(p).read_text() == f"a-{s}"
