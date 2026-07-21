# Understanding the statistics

mushin's statistical comparison layer is designed to give you honest answers:
not just *which method scored higher on average*, but *whether that difference
is reliable* given the seed-to-seed variance of training. This page explains the
tests, the Holm correction, and how to interpret the results.

## The tests

Pass `test=` to `compare` or `Study` to select the pairwise significance test:

| `test=` | Underlying scipy call | Paired? | When to use |
|---|---|---|---|
| `"wilcoxon"` | `scipy.stats.wilcoxon` | Yes | Default; non-normal distributions, ordinal metrics, small n |
| `"ttest_rel"` | `scipy.stats.ttest_rel` | Yes | Paired t-test; approximately normal data, equal variance assumed |
| `"welch"` | `scipy.stats.ttest_ind(equal_var=False)` | No | Gaussian metrics, unequal variance; good general choice |
| `"ttest_ind"` | `scipy.stats.ttest_ind(equal_var=True)` | No | Independent t-test, equal variance assumed |
| `"mannwhitney"` | `scipy.stats.mannwhitneyu` | No | Non-normal, independent samples |

**Paired vs independent:** Paired tests (`wilcoxon`, `ttest_rel`) compare
seed-matched values (seed 0 of method A vs seed 0 of method B). Use them when
the same seeds are used for both methods (the common case in mushin). Independent
tests (`welch`, `ttest_ind`, `mannwhitney`) treat the two groups as unrelated.

## Holm–Bonferroni correction

When you compare K methods on M metrics, mushin runs K×(K-1)/2 pairwise tests
per metric. Without correction, running many tests inflates the probability of a
false positive. mushin applies the **Holm–Bonferroni step-down correction** per
metric: it sorts the raw p-values, then adjusts the significance threshold for
each test in proportion to how many tests remain. The corrected p-values are
stored in `result.comparisons["p_corrected"]`.

The family-wise error rate is controlled at your chosen `alpha` (default 0.05).

## Effect size

In addition to the p-value, mushin reports **Cohen's d** as
`result.comparisons["effect_size"]`, matched to the test: paired tests
(`wilcoxon`, `ttest_rel`) report the paired *d<sub>z</sub>* — mean of the
per-seed differences over their standard deviation — while unpaired tests
report the classic pooled-variance d. Both measure the magnitude of the
difference in units of standard deviations:

| |d| | Interpretation |
|---|---|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| > 0.8 | Large |

A significant p-value with a small effect size means the difference is real but
may not matter in practice. A large effect size with a non-significant p-value
often means you have too few seeds.

## Single-seed behavior

With only one seed per method, there is no within-group variance. scipy tests
return a NaN p-value in this case. mushin treats NaN as *not significant* rather
than producing a false positive. NaN p-values are excluded from the Holm
correction so they cannot corrupt the correction of valid pairs.

## Underpowered-test warning

Some tests cannot reach a given `alpha` no matter how large the between-method
difference is, if the seed count is too low. For example, Wilcoxon over 3 seeds
has a best-case p-value of 0.25 — it can never reach the default `alpha=0.05`.
mushin warns you:

```
UserWarning: test='wilcoxon' cannot reach alpha=0.05 with 3 seeds
(best-case p=0.2500); use more seeds or a parametric test such as test='welch'.
```

**Solutions:**
- Switch to `test="welch"` (parametric; can reach significance with 3 seeds).
- Increase the number of seeds (5+ makes Wilcoxon viable).

## Interpreting the summary table

```python
result.summary()
# method | metric   | mean  | ci_low | ci_high | significant_vs_ref
# cnn    | accuracy | 0.963 | 0.951  | 0.975   |
# mlp    | accuracy | 0.941 | 0.928  | 0.954   | *
```

- `mean`: average metric value across seeds.
- `ci_low` / `ci_high`: 95% confidence interval (Student-t based).
- `significant_vs_ref`: `"*"` if the method differs significantly from the
  reference (first method listed) after Holm correction.

!!! tip "Pitfalls"
    - **Wilcoxon with few seeds:** Cannot reach p < 0.25 with 3 seeds. Use
      `test="welch"` instead.
    - **p-value vs effect size:** Statistical significance does not imply
      practical significance. Check `effect_size` alongside `p_corrected`.
    - **Single seed:** NaN p-value → not significant. Always use multiple seeds
      for meaningful comparisons.

## See also

- [Comparing methods](compare.md) — the `compare` API
- [API Reference — benchmark](../reference/benchmark.md)
