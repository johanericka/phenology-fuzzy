# Results Summary

Last updated: 2026-06-03

## Current Status

A one-scenario smoke test was run successfully with:

```bash
python3 main.py --test --no-plots
```

This verifies that the local simulation pipeline, controller variants, and AquaCrop integration run under the current environment. It does **not** provide final manuscript evidence.

## Smoke-Test Snapshot

For the single `2015 MT-2_Peralihan` test scenario:

| Controller | Mean IWU mm | AquaCrop transpiration stress days |
|---|---:|---:|
| Fuzzy-Phenology | 6.63 | 0.0 |
| Fuzzy-Static | 6.42 | 0.0 |
| Reactive-Static | 29.74 | 0.0 |
| Reactive-Phenology | 16.34 | 0.0 |

## Boundaries

- Treat current `output/` files as smoke-test outputs unless regenerated with the full reproducibility command.
- Do not use the smoke-test numbers for final water-saving, yield, IWUE, or statistical claims.
- `paired_ttest_results.csv` is empty in the smoke run.
- IWUE has `NaN` for low-irrigation fuzzy variants and needs investigation before becoming a headline metric.

## Detailed Audit

See `05_experiments/analysis/artifact-audit.md`.
