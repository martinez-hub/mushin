# llm

Evaluation of LLM systems, built on `compare_methods` with an item-level
bootstrap on top; the [LLM evaluation guide](../guides/llm.md) walks through
it. Needs the `eval` extra.

## Types

The shapes `compare_llms` accepts for its systems, examples and metrics, and
`llm_judge` returns a `Metric`. The signatures below spell them as `Any`, so
any callable or mapping of the right shape passes.

::: mushin.llm.System
    options:
      heading_level: 3

::: mushin.llm.Metric
    options:
      heading_level: 3

::: mushin.llm.Example
    options:
      heading_level: 3

::: mushin.llm.compare_llms

::: mushin.llm.compare_scores

::: mushin.llm.llm_judge
