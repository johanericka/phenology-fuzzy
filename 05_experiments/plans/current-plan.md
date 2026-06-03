# Current Experiment Plan

Last updated: 2026-06-03

## Current Status

The repository is runnable. A smoke test completed successfully, and the available metric families have been audited in `05_experiments/analysis/artifact-audit.md`.

## Canonical Commands From README

Main evaluation:

```bash
python main.py --reproducibility-pack   --reproducibility-start-year 2015   --reproducibility-end-year 2024   --experiment-mode all
```

Sensitivity checks:

```bash
python scripts/run_sensitivity_checks.py --with-aquacrop
```

## Completed Audit Step

Smoke test command:

```bash
python3 main.py --test --no-plots
```

Result: completed successfully. Current `output/` files are smoke-test artifacts and should be regenerated before manuscript use.

## Next Execution Decision

Recommended next run:

```bash
python3 main.py --reproducibility-pack   --reproducibility-start-year 2015   --reproducibility-end-year 2024   --experiment-mode final-evaluation
```

Use `--experiment-mode all` only if full tuning, final evaluation, attribution, and sensitivity outputs are desired in one run.

## Candidate Metrics For Final Analysis

- Total irrigation water use (IWU).
- Irrigation event count and depth.
- Target-band violation days, cumulative violation, and maximum violation.
- Yield or yield proxy if supported by AquaCrop/output artifacts.
- IWUE or water productivity, after investigating `NaN` behavior in low-irrigation cases.
- Seasonal robustness by MT-1, MT-2, and MT-3.
- Phenology-specific diagnostics for reproductive, flowering, and grain-filling stages.
