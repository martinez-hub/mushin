import pytest

from mushin.llm import llm_judge


def test_judge_scores_and_is_seeded():
    seen = {}

    def fake_judge(prompt, seed):
        seen["seed"] = seed
        return "yes" if "correct" in prompt else "no"

    metric = llm_judge(fake_judge, "Is this correct? correct", seed=7)
    assert metric("anything", "ref") == 1.0
    assert seen["seed"] == 7


def test_parse_variants():
    from mushin.llm._judge import parse_score

    assert parse_score("Yes, correct") == 1.0
    assert parse_score("no") == 0.0
    assert parse_score("score: 0.75") == 0.75
    assert parse_score("0.4 because ...") == 0.4
    with pytest.raises(ValueError, match="could not parse"):
        parse_score("hmm, maybe?")


def test_judge_metric_flows_through_compare_llms():
    from mushin.llm import compare_llms

    data = [{"input": i, "reference": "yes" if i % 2 == 0 else "no"} for i in range(6)]

    def judge(prompt, seed):
        # the template embeds the candidate output; score 1 if it contains "yes"
        return "yes" if "Candidate answer:\nyes" in prompt else "no"

    def sysA(inputs, seed):
        return ["yes"] * len(inputs)

    metric = llm_judge(judge, "Is the answer yes?")
    result = compare_llms({"A": sysA}, data, metric=metric, seeds=range(2))
    assert "score" in result.data.data_vars


def test_parse_rejects_out_of_range():
    from mushin.llm._judge import parse_score

    with pytest.raises(ValueError, match="outside"):
        parse_score("score: 5")
    with pytest.raises(ValueError, match="outside"):
        parse_score("7")


def test_judge_receives_trial_seed():
    """compare_llms threads the per-trial seed through an llm_judge metric."""
    from mushin.llm import compare_llms

    seen = set()

    def judge(prompt, seed):
        seen.add(seed)
        return "yes"

    data = [{"input": i, "reference": "x"} for i in range(3)]

    def sysA(inputs, seed):
        return ["yes"] * len(inputs)

    compare_llms({"A": sysA}, data, metric=llm_judge(judge, "ok?"), seeds=[5, 9])
    assert seen == {5, 9}  # judge got the trial seeds, not the fixed default 0


def test_parse_yes_no_requires_word_boundary():
    from mushin.llm._judge import parse_score

    assert parse_score("yes") == 1.0
    assert parse_score("no") == 0.0
    assert parse_score("Yes, definitely") == 1.0
    with pytest.raises(ValueError):  # "nope" must not match "no"
        parse_score("nope")
    with pytest.raises(ValueError):  # "yesterday" must not match "yes"
        parse_score("yesterday's answer is better")
