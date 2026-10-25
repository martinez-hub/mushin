# llm

Evaluation of LLM systems, built on `compare_methods` with an item-level
bootstrap on top; the [LLM evaluation guide](../guides/llm.md) walks through
it. Needs the `eval` extra.

## Types

Three aliases name what `compare_llms` takes; `llm_judge` returns a `Metric`:

- `System`: a callable `(inputs, seed) -> outputs`. It maps a batch of inputs
  and an integer seed to a batch of outputs in the same order.
- `Metric`: a `torchmetrics.Metric`, or a per-example scorer
  `(output, reference) -> float`.
- `Example`: one evaluation example, either a mapping
  `{"input": ..., "reference": ...}` or a bare input when the metric needs no
  reference. A bare tuple is one input, not an `(input, reference)` pair; use
  the mapping form to attach a reference.

::: mushin.llm.compare_llms

::: mushin.llm.compare_scores

::: mushin.llm.llm_judge
