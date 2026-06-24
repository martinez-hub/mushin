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
