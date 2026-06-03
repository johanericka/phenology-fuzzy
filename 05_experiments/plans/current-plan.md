# Current Experiment Plan

## Current Status

Existing code is present and appears to support a reproducibility package and sensitivity checks. The first research-assistant task is to audit what the code actually runs and what output artifacts already exist.

## Canonical Commands From README

Main evaluation:

```bash
python main.py --reproducibility-pack   --reproducibility-start-year 2015   --reproducibility-end-year 2024   --experiment-mode all
```

Sensitivity checks:

```bash
python scripts/run_sensitivity_checks.py --with-aquacrop
```

## Planned Audit

1. Verify Python environment and dependencies.
2. Run a small smoke test if available.
3. Inventory existing `output/` artifacts without treating them as final until checked.
4. Identify metrics produced by the pipeline.
5. Map each metric to manuscript-safe claims.

## Candidate Metrics

- Total irrigation water use (IWU).
- Irrigation event count and depth.
- Target-band violation days, cumulative violation, and maximum violation.
- Yield or yield proxy if supported by AquaCrop/output artifacts.
- IWUE or water productivity if yield and water use are available.
- Seasonal robustness by MT-1, MT-2, and MT-3.
