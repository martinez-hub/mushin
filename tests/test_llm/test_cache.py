def test_partition_and_put(tmp_path):
    from mushin.llm._cache import OutputCache

    c = OutputCache(tmp_path)
    inputs = ["a", "b", "c"]
    cached, missing = c.partition("sys", 0, inputs)
    assert cached == {} and [m[1] for m in missing] == inputs

    c.put_many("sys", 0, [("a", "A"), ("c", "C")])
    cached, missing = c.partition("sys", 0, inputs)
    assert cached == {0: "A", 2: "C"}
    assert [m for m in missing] == [(1, "b")]  # only the uncached one
