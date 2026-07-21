"""Compare two fake LLM systems with statistical significance.

Run it (no network, no real LLM):  python examples/compare_llms_demo.py
Requires the eval extra:  pip install "mushin-py[eval]"

The reusable `run(data)` core is exercised by the test suite on a small
synthetic eval set, so CI never needs a real model or API key.
"""

from __future__ import annotations

import random

from mushin.benchmark import BenchmarkResult
from mushin.llm import compare_llms


# --8<-- [start:run]
def _exact_match(output: str, reference: str) -> float:
    return float(output.strip() == reference.strip())


def run(data: list[dict]) -> BenchmarkResult:
    """Compare a strong system against a biased one on `data`.

    Both systems are fakes — no network, no real model. Each wires the trial
    `seed` to a little stochasticity (the way a real temperature>0 model would),
    so the per-seed scores form a genuine sampling distribution and the
    significance test is meaningful. Replace them with your own callables (or
    hydra-zen configs) to run a real evaluation.
    """

    def strong(inputs, seed):
        """Mostly correct, with a small seed-driven error rate."""
        rng = random.Random(seed)
        out = []
        for i in inputs:
            label = "even" if i % 2 == 0 else "odd"
            if rng.random() < 0.15:  # occasional slip -> score varies across seeds
                label = "odd" if label == "even" else "even"
            out.append(label)
        return out

    def biased(inputs, seed):
        """Leans toward 'even', with a little seed-driven noise of its own."""
        rng = random.Random(1000 + seed)
        return ["even" if rng.random() < 0.85 else "odd" for _ in inputs]

    return compare_llms(
        systems={"strong": strong, "biased": biased},
        data=data,
        metric=_exact_match,
        seeds=range(5),
        test="welch",
    )


# --8<-- [end:run]


def _build_data(n: int = 20) -> list[dict]:
    return [
        {"input": i, "reference": "even" if i % 2 == 0 else "odd"} for i in range(n)
    ]


if __name__ == "__main__":
    data = _build_data()
    result = run(data)
    print(result.summary().to_string(index=False))
