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
