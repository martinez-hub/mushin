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


def test_key_is_type_preserving_and_injective():
    """Distinct Python inputs that JSON would conflate must hash differently."""
    from mushin.llm._cache import _key

    assert _key({1: "x"}) != _key({"1": "x"})  # int key vs str key
    assert _key(("a", 1)) != _key(["a", 1])  # tuple vs list
    # but equal values (any dict order) hash the same -> cache still hits
    assert _key({"a": 1, "b": 2}) == _key({"b": 2, "a": 1})


def test_distinct_structured_inputs_do_not_collide(tmp_path):
    """A cached output for one structured input must not be served for a different
    one that only JSON would treat as equal."""
    from mushin.llm._cache import OutputCache

    c = OutputCache(tmp_path)
    c.put_many("sys", 0, [({1: "x"}, "A")])  # cache an int-keyed input
    cached, missing = c.partition("sys", 0, [{1: "x"}, {"1": "x"}])
    assert cached == {0: "A"}  # the int-keyed input hits
    assert [m[1] for m in missing] == [{"1": "x"}]  # str-keyed input is a MISS, not "A"


def test_cache_path_stays_under_root(tmp_path):
    from mushin.llm._cache import OutputCache

    c = OutputCache(tmp_path)
    c.put_many("../evil/sys", 0, [("a", "A")])  # path-traversal-looking name
    written = list(tmp_path.rglob("*.jsonl"))
    assert written and all(str(p).startswith(str(tmp_path)) for p in written)
    cached, _ = c.partition("../evil/sys", 0, ["a"])
    assert cached == {0: "A"}  # still round-trips


def test_truncated_trailing_line_is_skipped_not_fatal(tmp_path):
    """A partial line left by a crash mid-write must not poison the whole cache:
    good records still load and the missing input is simply recomputed."""
    from mushin.llm._cache import OutputCache

    c = OutputCache(tmp_path)
    c.put_many("sys", 0, [("a", "A"), ("b", "B")])
    path = c._path("sys", 0)
    with path.open("a") as f:
        f.write('{"key": "deadbeef", "output": "partial')  # truncated, no newline

    cached, missing = c.partition("sys", 0, ["a", "b", "c"])
    assert cached == {0: "A", 1: "B"}  # the two good records still load
    assert [m[1] for m in missing] == ["c"]  # corrupt/absent -> recompute
