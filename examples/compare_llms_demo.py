"""Compare two fake LLM systems with statistical significance.

Run it (no network, no real LLM):  python examples/compare_llms_demo.py

The reusable `run(data)` core is exercised by the test suite on a small
synthetic eval set, so CI never needs a real model or API key.
"""

from __future__ import annotations

from mushin.benchmark import BenchmarkResult
from mushin.llm import compare_llms


# --8<-- [start:run]
def _exact_match(output: str, reference: str) -> float:
    return float(output.strip() == reference.strip())


def run(data: list[dict]) -> BenchmarkResult:
    """Compare a 'perfect' system against a 'biased' system on `data`.

    Both systems are deterministic fakes — no network, no real model.
    Replace them with your own callables (or hydra-zen configs) to run
    a real evaluation.
    """

    def perfect(inputs, seed):
        """Always returns the correct answer (even / odd label)."""
        return ["even" if i % 2 == 0 else "odd" for i in inputs]

    def biased(inputs, seed):
        """Always guesses 'even', regardless of the input."""
        return ["even"] * len(inputs)

    return compare_llms(
        systems={"perfect": perfect, "biased": biased},
        data=data,
        metric=_exact_match,
        seeds=range(5),
        test="welch",
    )


# --8<-- [end:run]


def _build_data(n: int = 20) -> list[dict]:
    return [{"input": i, "reference": "even" if i % 2 == 0 else "odd"} for i in range(n)]


if __name__ == "__main__":
    data = _build_data()
    result = run(data)
    print(result.summary().to_string(index=False))
