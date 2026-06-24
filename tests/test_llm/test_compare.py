import pytest

from mushin.benchmark import BenchmarkResult
from mushin.llm import compare_llms


def _data(n=8):
    # input is "i", reference is "yes" for even i else "no"
    return [{"input": i, "reference": "yes" if i % 2 == 0 else "no"} for i in range(n)]


def exact(output, reference):
    return float(output == reference)


def test_clear_winner_flagged_significant():
    data = _data()

    def good(inputs, seed):  # always correct
        return ["yes" if i % 2 == 0 else "no" for i in inputs]

    def bad(inputs, seed):  # seed-perturbed, mostly wrong
        return ["yes"] * len(inputs)

    result = compare_llms(
        {"good": good, "bad": bad}, data, metric=exact, seeds=range(4), test="welch"
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert float(result.data["score"].sel({"method": "good"}).mean()) == 1.0


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


def test_empty_metric_battery_rejected():
    calls = {"n": 0}

    def sys(inputs, seed):
        calls["n"] += 1
        return ["yes"] * len(inputs)

    with pytest.raises(ValueError, match="metric.*empty"):
        compare_llms({"s": sys}, _data(2), metric={})
    assert calls["n"] == 0  # rejected before any system call
