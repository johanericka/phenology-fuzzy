# Artifact Audit

Last updated: 2026-06-03

## Scope

This audit initializes the existing repository as a research workspace and checks the local runnable artifacts without making scientific claims beyond what was directly observed.

## Repository State

- Workspace: `/home/johanericka/phenology-fuzzy`
- Branch: `main`
- Latest research-workspace commit before this audit: `afc3d5a Initialize research assistant workspace`
- Pre-existing dirty file not touched by this audit: `.gitignore`
- Canonical code remains at the repository root and `src/`; it has not been migrated into `05_experiments/code/`.

## Environment Check

Observed with `python3`:

| Package | Status |
|---|---|
| Python | 3.13.9 |
| numpy | OK, 2.4.4 |
| pandas | OK, 2.3.3 |
| matplotlib | OK, 3.10.7 |
| scipy | OK, 1.17.1 |
| seaborn | OK, 0.13.2 |
| aquacrop | OK, version not reported |
| skfuzzy | OK, 0.5.0 |

`main.py --help` works and exposes these research-relevant modes:

- `--test`
- `--season {MT-1,MT-2,MT-3}`
- `--reproducibility-pack`
- `--reproducibility-start-year`
- `--reproducibility-end-year`
- `--experiment-mode {all,tuning,final-evaluation,attribution,sensitivity}`

## Smoke Test Run

Command executed:

```bash
python3 main.py --test --no-plots
```

Result: completed successfully.

Important boundary: this was a one-scenario smoke test, not the final reproducibility evaluation. It wrote test-mode outputs into `output/`, so files currently under `output/` should be treated as smoke-test artifacts unless regenerated with the full reproducibility command.

Smoke-test setup observed from console:

- Years filtered to 2015-2016 for weather loading.
- Test mode season filter: `MT-2` / `MT-2_Peralihan`.
- Tuning skipped in `--test` mode.
- One scenario was completed for each controller family.
- AquaCrop yield/stress integration ran for each controller.

## Smoke-Test Output Inventory

Current `output/` files include:

- `fuzzy_phenology_results.csv`
- `fuzzy_phenology_summary.csv`
- `fuzzy_static_results.csv`
- `fuzzy_static_summary.csv`
- `reactive_static_results.csv`
- `reactive_static_summary.csv`
- `reactive_stage_results.csv`
- `reactive_stage_summary.csv`
- `controller_overall_summary.csv`
- `controller_seasonal_summary.csv`
- `head_to_head_per_skenario_2015_2015.csv`
- `head_to_head_ringkasan_per_musim_2015_2015.csv`
- `mechanism_diagnostics_per_scenario.csv`
- `mechanism_diagnostics_summary.csv`
- `attribution_isolation_table.csv`
- `controller_design_summary.csv`
- `analisa.md`
- `analisa_peralihan.md`
- `phenology_targets.png`
- `fuzzy_mf_visualization.png`
- `paired_ttest_results.csv` exists but is effectively empty in the smoke run.

## Metrics Present in Outputs

The result files include these metric families:

| Metric family | Example columns | Manuscript relevance |
|---|---|---|
| Irrigation water use | `iwu_mm`, `mean_iwu_mm` | Primary water-use comparison |
| Irrigation behavior | `n_irrigation_events`, `mean_depth_per_event_mm`, `irrigation_frequency_pct`, `max_interval_without_irrigation_days` | Controller behavior and interpretability |
| Target tracking | `target_pct`, `days_in_target`, `stress_days`, `excess_days`, `mse`, `r_squared` | Dynamic-band control quality |
| Soil moisture state | `r`, `r_pct`, `theta`, `theta_prev`, `mean_r_pct` | State trajectory and diagnostics |
| Weather/water balance | `precipitation`, `et0`, `etc`, `runoff`, `deep_percolation` | Process explanation |
| AquaCrop/yield diagnostics | `yield_dry_t_ha`, `iwue_kg_ha_per_mm`, `aq_transpiration_ratio`, `aq_stress_days_tr` | Agronomic and water-productivity outcomes |
| Stage-specific AquaCrop diagnostics | reproductive, flowering, grain-fill stress/irrigation fields | Phenology-specific mechanism support |

## Smoke-Test Numerical Snapshot

From `controller_overall_summary.csv` for the one MT-2 scenario:

| Controller | n | Mean IWU mm | Mean AquaCrop transpiration stress days |
|---|---:|---:|---:|
| Fuzzy-Phenology | 1 | 6.63 | 0.0 |
| Fuzzy-Static | 1 | 6.42 | 0.0 |
| Reactive-Static | 1 | 29.74 | 0.0 |
| Reactive-Phenology | 1 | 16.34 | 0.0 |

Interpretation boundary: this snapshot only verifies that the metric pipeline runs. It is not enough to claim controller superiority because it uses one season-year scenario and `--test` skips tuning.

## Claim-Safety Assessment

Safe now:

- The repository contains a runnable closed-loop irrigation simulation with phenology-aware fuzzy, fuzzy-static, reactive-static, and reactive-phenology controller variants.
- The pipeline can produce irrigation, target-tracking, soil-moisture, AquaCrop/yield, and phenology-specific mechanism diagnostics.
- The smoke test runs under the current Python environment.

Not safe yet:

- Final water-saving claims.
- Final IWUE/yield improvement claims.
- Statistical significance claims.
- General claims across 2015-2024 or across all seasons.
- Novelty claims versus literature.

## Issues and Risks

1. `output/` currently reflects the smoke test and should be regenerated for full evaluation before manuscript use.
2. `paired_ttest_results.csv` is empty in the smoke run, which is expected for insufficient paired samples but must not be used as evidence.
3. IWUE is `NaN` for low-irrigation fuzzy variants in the smoke snapshot; this needs investigation before using IWUE as a headline metric.
4. Literature support for dynamic target bands is not yet audited in this workspace.
5. `.gitignore` remains dirty from a pre-existing change outside this audit.

## Recommended Next Step

Run the full reproducibility pack after confirming runtime cost is acceptable:

```bash
python3 main.py --reproducibility-pack   --reproducibility-start-year 2015   --reproducibility-end-year 2024   --experiment-mode all
```

If runtime is too long, run one bounded protocol first:

```bash
python3 main.py --reproducibility-pack   --reproducibility-start-year 2015   --reproducibility-end-year 2024   --experiment-mode final-evaluation
```

After full outputs exist, update this audit into a results-readiness memo and then proceed to RAG-first SOTA/gap validation.
