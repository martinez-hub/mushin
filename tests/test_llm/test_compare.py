import pytest

from mushin.benchmark import BenchmarkResult
from mushin.llm import compare_llms


def _data(n=8):
    # input is "i", reference is "yes" for even i else "no"
    return [{"input": i, "reference": "yes" if i % 2 == 0 else "no"} for i in range(n)]


def exact(output, reference):
    return float(output == reference)


def test_clear_winner_flagged_significant():
    import random

    n = 12
    data = _data(n)
    truth = ["yes" if i % 2 == 0 else "no" for i in range(n)]

    def good(inputs, seed):  # ~85% correct, varies by seed (deterministic per seed)
        rng = random.Random(seed)
        return [truth[i] if rng.random() > 0.15 else "no" for i in inputs]

    def bad(inputs, seed):  # ~50% correct, varies by seed
        rng = random.Random(seed + 991)
        return [truth[i] if rng.random() > 0.5 else "no" for i in inputs]

    result = compare_llms(
        {"good": good, "bad": bad}, data, metric=exact, seeds=range(5), test="welch"
    )
    assert isinstance(result, BenchmarkResult)
    # The clear winner must actually be FLAGGED significant (the point of the
    # test): both systems have real across-seed variance, so the comparison is
    # not masked, and the ~0.35 gap is significant at 5 seeds under Welch.
    row = result.comparisons[result.comparisons["metric"] == "score"].iloc[0]
    assert bool(row["significant"])
    assert abs(float(row["mean_diff"])) > 0.15


def test_seed_controlled_variance_and_reproducible():
    data = _data()

    def stochastic(inputs, seed):
        # seed flips one example's answer -> small reproducible variance
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[seed % len(out)] = "no"
        return out

    r1 = compare_llms({"s": stochastic}, data, metric=exact, seeds=range(3))
    r2 = compare_llms({"s": stochastic}, data, metric=exact, seeds=range(3))
    assert r1.data["score"].values.tolist() == r2.data["score"].values.tolist()


def test_torchmetrics_scalar_and_dict_battery():
    from torchmetrics.text import WordErrorRate

    class FakeDictMetric(WordErrorRate):
        # returns a dict from compute() to exercise expansion
        def compute(self):
            v = float(super().compute())
            return {"a": v, "b": 1.0 - v}

    data = [{"input": i, "reference": "the cat"} for i in range(4)]

    def sysA(inputs, seed):
        return ["the cat"] * len(inputs)

    result = compare_llms(
        {"A": sysA},
        data,
        metric={"wer": WordErrorRate(), "split": FakeDictMetric()},
        seeds=range(2),
    )
    assert "wer" in result.data.data_vars
    assert "split_a" in result.data.data_vars and "split_b" in result.data.data_vars


def test_per_example_dict_metric_is_averaged():
    """A dict metric whose values are per-example tensors (e.g. BERTScore returns
    per-prediction precision/recall/f1) is averaged over examples instead of
    raising on float() of a multi-element tensor."""
    import numpy as np
    import torch
    from torchmetrics.text import WordErrorRate

    class PerExampleMetric(WordErrorRate):
        def update(self, preds, target):
            self._n = len(list(preds))

        def compute(self):
            return {
                "precision": torch.linspace(0.0, 1.0, self._n),  # per-example vector
                "recall": torch.full((self._n,), 0.5),
            }

    data = [{"input": i, "reference": "x"} for i in range(4)]

    def sysA(inputs, seed):
        return ["y"] * len(inputs)

    result = compare_llms(
        {"A": sysA}, data, metric={"bert": PerExampleMetric()}, seeds=range(2)
    )
    assert "bert_precision" in result.data.data_vars
    assert "bert_recall" in result.data.data_vars
    # averaged over the 4 examples: mean(linspace(0,1,4)) == 0.5, mean(0.5...) == 0.5
    assert np.isclose(float(result.data["bert_precision"].mean()), 0.5)
    assert np.isclose(float(result.data["bert_recall"].mean()), 0.5)


def test_cache_skips_recalls(tmp_path):
    data = _data(4)
    calls = {"n": 0}

    def counting(inputs, seed):
        calls["n"] += len(inputs)
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    compare_llms({"c": counting}, data, metric=exact, seeds=range(2), cache=tmp_path)
    first = calls["n"]
    compare_llms({"c": counting}, data, metric=exact, seeds=range(2), cache=tmp_path)
    assert calls["n"] == first  # second run fully cached -> no new calls


def test_empty_inputs_rejected():
    with pytest.raises(ValueError):
        compare_llms({"a": lambda i, s: []}, [], metric=exact)


def test_real_squad_dict_metric_expands():
    """A real dict-returning torchmetrics metric works when output/reference are
    shaped to its update() contract, and expands to one data variable per key."""
    from torchmetrics.text import SQuAD

    data = [
        {
            "input": i,
            "reference": {
                "answers": {"answer_start": [0], "text": ["cat"]},
                "id": str(i),
            },
        }
        for i in range(4)
    ]

    def sysA(inputs, seed):
        return [{"prediction_text": "cat", "id": str(i)} for i in inputs]

    result = compare_llms({"A": sysA}, data, metric=SQuAD(), seeds=range(2))
    # single dict metric -> bare subkeys
    assert "exact_match" in result.data.data_vars
    assert "f1" in result.data.data_vars
    assert float(result.data["exact_match"].sel({"method": "A"}).mean()) == 100.0


def test_cache_rejects_non_serializable_output(tmp_path):
    data = _data(2)

    class NotJSON:  # a non-serializable system output
        pass

    def weird(inputs, seed):
        return [NotJSON() for _ in inputs]

    with pytest.raises(TypeError, match="JSON-serializable"):
        compare_llms({"w": weird}, data, metric=exact, seeds=range(1), cache=tmp_path)


def test_warns_on_deterministic_zero_variance_system():
    """A system that ignores the seed yields identical scores across seeds; the
    seeds are duplicated points, so mushin warns that significance is unreliable."""
    data = _data(6)

    def deterministic(inputs, seed):  # ignores seed entirely
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    def other(inputs, seed):
        return ["yes"] * len(inputs)

    with pytest.warns(UserWarning, match="identical scores across all"):
        result = compare_llms(
            {"a": deterministic, "b": other}, data, metric=exact, seeds=range(4)
        )
    # zero-variance systems must NOT be reported significant (no duplicated-point
    # false positive), and the p-value is NaN.
    assert not result.comparisons["significant"].any()
    assert result.comparisons["p_value"].isna().all()


def test_actual_seed_values_preserved():
    data = _data(4)

    def sysA(inputs, seed):
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    result = compare_llms({"A": sysA}, data, metric=exact, seeds=[13, 21])
    assert result.data.coords["seed"].values.tolist() == [13, 21]


def test_zero_variance_masked_per_metric():
    """Masking is per-metric: a metric that's constant across seeds is masked,
    while a metric that varies in the same run is left as a real test."""
    import warnings

    data = _data(6)

    def A(inputs, seed):
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[seed % len(out)] = "no"  # varies with seed
        return out

    def B(inputs, seed):
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[(seed + 1) % len(out)] = "yes"  # varies differently
        return out

    def const(o, r):
        return 1.0  # constant for every example/seed -> zero variance

    metric = {"const": const, "acc": exact}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare_llms({"A": A, "B": B}, data, metric=metric, seeds=range(4))

    comps = result.comparisons
    const_row = comps[comps["metric"] == "const"].iloc[0]
    acc_row = comps[comps["metric"] == "acc"].iloc[0]
    assert const_row["p_value"] != const_row["p_value"]  # NaN -> masked
    assert acc_row["p_value"] == acc_row["p_value"]  # real p-value -> not masked


def test_holm_recorrected_over_surviving_pairs():
    """When a zero-variance system is masked, the remaining valid pair is
    Holm-corrected over the reduced family (not over-corrected)."""
    import warnings

    data = _data(8)

    def A(inputs, seed):
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[seed % len(out)] = "no"
        return out

    def B(inputs, seed):
        out = ["yes" if i % 3 == 0 else "no" for i in inputs]
        out[(seed + 1) % len(out)] = "yes"
        return out

    def C(inputs, seed):  # constant -> zero variance in `score`
        return ["yes"] * len(inputs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compare_llms({"A": A, "B": B, "C": C}, data, metric=exact, seeds=range(5))

    comps = r.comparisons
    ab = comps[
        ((comps.method_a == "A") & (comps.method_b == "B"))
        | ((comps.method_a == "B") & (comps.method_b == "A"))
    ].iloc[0]
    # A-B is the only surviving pair -> Holm family of 1 -> no inflation
    assert ab["p_corrected"] == ab["p_value"]
    # every comparison involving C is masked
    cpairs = comps[(comps.method_a == "C") | (comps.method_b == "C")]
    assert cpairs["p_value"].isna().all()


def test_empty_seeds_rejected():
    with pytest.raises(ValueError, match="seed"):
        compare_llms({"a": lambda ins, s: list(ins)}, _data(2), metric=exact, seeds=[])


def test_empty_seeds_rejected_before_instantiating_systems():
    """The empty-seeds guard fires before `as_system` instantiates configs, so a
    hydra-zen system (which may load a large model) is not built just to fail."""
    # A config whose _target_ cannot be imported: if instantiation ran first we'd
    # get an import error, not the seeds ValueError.
    bad_config = {"_target_": "this.module.does.not.exist"}
    with pytest.raises(ValueError, match="seed"):
        compare_llms({"a": bad_config}, _data(2), metric=exact, seeds=[])


def test_duplicate_seeds_rejected_before_running_systems():
    """Duplicate seeds are the same trial; counting them as independent samples
    would inflate significance, so they are rejected before any system call."""
    calls = {"n": 0}

    def sys(inputs, seed):
        calls["n"] += 1
        return ["yes"] * len(inputs)

    with pytest.raises(ValueError, match="duplicate"):
        compare_llms({"s": sys}, _data(2), metric=exact, seeds=[0, 0, 1])
    assert calls["n"] == 0  # rejected before any (token-spending) system call


def test_cache_normalizes_output_json_form(tmp_path):
    """A fresh cached run scores the JSON-normalized output (tuple -> list), so it
    matches what a later cache replay scores."""
    data = _data(3)

    def sys(inputs, seed):
        return [("a", i) for i in inputs]  # tuples (JSON-encoded as lists)

    def is_list(output, reference):
        return float(isinstance(output, list))

    r1 = compare_llms({"s": sys}, data, metric=is_list, seeds=range(1), cache=tmp_path)
    r2 = compare_llms({"s": sys}, data, metric=is_list, seeds=range(1), cache=tmp_path)
    assert float(r1.data["score"].mean()) == 1.0  # fresh run already normalized
    assert r1.data["score"].values.tolist() == r2.data["score"].values.tolist()


def test_unknown_test_rejected_before_running_systems():
    calls = {"n": 0}

    def sys(inputs, seed):
        calls["n"] += 1
        return ["yes"] * len(inputs)

    with pytest.raises(ValueError, match="unknown test"):
        compare_llms({"s": sys}, _data(2), metric=exact, test="bogus")
    assert calls["n"] == 0  # validated before any (token-spending) system call


def test_colliding_metric_output_names_rejected():
    """A dict metric that expands to a name already used by another battery entry
    must raise, not silently overwrite (which would report the wrong metric)."""
    from torchmetrics.text import WordErrorRate

    class FakeDictMetric(WordErrorRate):
        def compute(self):
            v = float(super().compute())
            return {"a": v, "b": 1.0 - v}  # -> split_a / split_b

    data = [{"input": i, "reference": "the cat"} for i in range(4)]

    def sysA(inputs, seed):
        return ["the cat"] * len(inputs)

    def collide(o, r):
        return 1.0

    # "split" expands to split_a/split_b; the second entry is also named "split_a".
    metric = {"split": FakeDictMetric(), "split_a": collide}
    with pytest.raises(ValueError, match="colliding"):
        compare_llms({"A": sysA}, data, metric=metric, seeds=range(2))


def test_empty_metric_battery_rejected():
    calls = {"n": 0}

    def sys(inputs, seed):
        calls["n"] += 1
        return ["yes"] * len(inputs)

    with pytest.raises(ValueError, match="metric.*empty"):
        compare_llms({"s": sys}, _data(2), metric={})
    assert calls["n"] == 0  # rejected before any system call


def test_score_is_cache_independent_for_structured_outputs(tmp_path):
    """A structured (non-string) output scores the same with and without `cache=`;
    the cache is an optimization, not a result-changing switch."""

    def sys(inputs, seed):
        return [("out", i) for i in inputs]  # tuples -> JSON-normalized to lists

    def is_list(output, reference):
        return float(isinstance(output, list))

    data = _data(3)
    no_cache = compare_llms({"s": sys}, data, metric=is_list, seeds=range(1))
    cached = compare_llms(
        {"s": sys}, data, metric=is_list, seeds=range(1), cache=tmp_path
    )
    assert (
        no_cache.data["score"].values.tolist() == cached.data["score"].values.tolist()
    )
    assert float(no_cache.data["score"].mean()) == 1.0  # both see a list, not a tuple


def test_structured_output_matches_identical_structured_reference():
    """Output and reference are normalized on equal footing, so a structured label
    (e.g. a tuple) matches an identical output instead of failing because the
    output was JSON-normalized to a list while the reference stayed a tuple."""
    data = [
        {"input": i, "reference": ("even" if i % 2 == 0 else "odd", i)}
        for i in range(4)
    ]

    def sysA(inputs, seed):
        return [("even" if i % 2 == 0 else "odd", i) for i in inputs]

    def exact(output, reference):
        return float(output == reference)

    r = compare_llms({"A": sysA}, data, metric=exact, seeds=range(2))
    assert float(r.data["score"].mean()) == 1.0  # both round-trip to lists -> equal


def test_sub_epsilon_within_group_variance_is_masked():
    """A system whose seed-to-seed scores differ only at sub-epsilon scale has no
    real sampling distribution, so it is masked rather than reported significant
    via catastrophic cancellation."""
    data = _data(4)

    def sysA(inputs, seed):
        return ["x"] * len(inputs)

    def sysB(inputs, seed):
        return ["y"] * len(inputs)

    def metric(output, reference, seed=0):
        # near-constant per system: ~1e-9 jitter is within np.allclose, so there is
        # no meaningful variance, but the two systems' means differ widely.
        base = 0.5 if output == "x" else 0.9
        return base + seed * 1e-9

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare_llms(
            {"A": sysA, "B": sysB}, data, metric=metric, seeds=range(4)
        )
    # both systems are effectively constant -> every comparison masked, not "significant"
    assert result.comparisons["p_value"].isna().all()
    assert not result.comparisons["significant"].any()


def test_masked_rows_have_nan_effect_size():
    """A masked (no-sampling-distribution) row reports NaN effect_size rather than
    a meaningless ±inf/huge standardized effect next to significant=False."""
    import warnings

    data = _data(6)

    def const_a(inputs, seed):  # constant: 0.5 accuracy
        return ["yes"] * len(inputs)

    def const_b(inputs, seed):  # constant: 1.0 accuracy (different mean)
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compare_llms(
            {"a": const_a, "b": const_b}, data, metric=exact, seeds=range(3)
        )
    assert r.comparisons["effect_size"].isna().all()
    assert r.comparisons["p_value"].isna().all()
    assert not r.comparisons["significant"].any()


def test_system_receives_actual_seed_values():
    """The system callable is invoked with the requested seed values (not 0..n-1)."""
    data = _data(3)
    seen = set()

    def sysA(inputs, seed):
        seen.add(seed)
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    compare_llms({"A": sysA}, data, metric=exact, seeds=[5, 9])
    assert seen == {5, 9}


def test_per_metric_masking_does_not_warn_whole_system():
    """Masking a single constant metric must NOT emit the whole-system
    'identical scores' warning, which is reserved for systems constant in *every*
    metric."""
    import warnings

    data = _data(6)

    def A(inputs, seed):
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[seed % len(out)] = "no"  # `acc` varies with seed
        return out

    def B(inputs, seed):
        out = ["yes" if i % 2 == 0 else "no" for i in inputs]
        out[(seed + 1) % len(out)] = "yes"
        return out

    def const(o, r):
        return 1.0  # constant only in this metric

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compare_llms(
            {"A": A, "B": B},
            data,
            metric={"const": const, "acc": exact},
            seeds=range(4),
        )
    msgs = [str(w.message) for w in caught]
    assert not any("identical scores across all" in m for m in msgs)


def test_surviving_pairs_recorrected_over_reduced_family():
    """With a masked zero-variance system, the surviving pairs are Holm-corrected
    over the reduced family (not the original, larger family), and `significant`
    is derived from those re-corrected p-values."""
    import warnings

    import numpy as np

    from mushin.benchmark._stats import holm_correction

    data = _data(4)
    # Three varying systems (A, B, D) + one constant (C, masked). Per-seed scores
    # are set directly via a seed-aware metric so the p-values are controlled.
    # These are tuned so B-D is NON-significant under the original family of 6
    # (p_corrected = 0.078) but SIGNIFICANT after re-Holm over the 3 survivors
    # (0.039) — i.e. the re-correction genuinely flips the result, so a regression
    # that dropped it would be caught here.
    means = {
        "A": [0.84, 0.87, 0.90, 0.93, 0.96],
        "B": [0.575, 0.6075, 0.64, 0.6725, 0.705],
        "D": [0.495, 0.5275, 0.56, 0.5925, 0.625],
        "C": [0.68, 0.68, 0.68, 0.68, 0.68],  # constant -> masked
    }

    def make(name):
        def s(inputs, seed):
            return [name] * len(inputs)

        return s

    def metric(output, reference, seed=0):
        return means[output][seed]

    systems = {name: make(name) for name in means}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compare_llms(systems, data, metric=metric, seeds=range(5))

    comps = r.comparisons
    involves_c = (comps["method_a"] == "C") | (comps["method_b"] == "C")
    survivors = comps[~involves_c]
    masked = comps[involves_c]

    assert masked["p_value"].isna().all()  # every C pair masked
    assert len(survivors) == 3  # A-B, A-D, B-D all survive

    # p_corrected must equal Holm over ONLY the survivors' raw p-values (a family of
    # 3), not the original family of 6 that compare_methods first corrected over.
    expected = holm_correction(survivors["p_value"].tolist())
    assert np.allclose(sorted(survivors["p_corrected"].tolist()), sorted(expected))
    # and significant is derived from the re-corrected p-values
    for _, row in survivors.iterrows():
        assert row["significant"] == bool(row["p_corrected"] < r.alpha)

    # B-D is the load-bearing case: significant ONLY because of the re-correction
    # (under the original family of 6 it is p_corrected≈0.078, not significant).
    bd = survivors[
        ((survivors.method_a == "B") & (survivors.method_b == "D"))
        | ((survivors.method_a == "D") & (survivors.method_b == "B"))
    ].iloc[0]
    assert bd["significant"] and bd["p_corrected"] < r.alpha


def test_dict_without_input_key_is_treated_as_bare_input():
    from mushin.llm._compare import _normalize_examples

    inputs, refs = _normalize_examples(
        [{"prompt": "hi"}, {"input": "x", "reference": "y"}]
    )
    assert inputs == [{"prompt": "hi"}, "x"]  # bare dict input vs example wrapper
    assert refs == [None, "y"]


def test_bare_two_tuple_is_an_input_not_split_into_reference():
    """A 2-tuple is a bare input kept intact (e.g. (system_prompt, user_prompt)),
    NOT split into (input, reference) — references come only via the dict form."""
    from mushin.llm._compare import _normalize_examples

    inputs, refs = _normalize_examples(
        [("system", "user"), {"input": "x", "reference": "y"}]
    )
    assert inputs == [("system", "user"), "x"]  # tuple passed through whole
    assert refs == [None, "y"]
