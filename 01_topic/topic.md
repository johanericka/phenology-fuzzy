# Topic Statement

## Working Title

Phenology-Aware Fuzzy Irrigation Control for Paddy Rice Using Dynamic Soil-Moisture Target Bands

## Problem

Static irrigation thresholds can be scientifically weak for rice because crop water sensitivity changes across phenological stages. This project tests a controller that adapts soil-moisture target bands by growth stage and compares it against static/reactive baselines under weather-driven closed-loop simulation.

## Existing Repository Assets

- `main.py`: main simulation and reproducibility entry point.
- `src/`: phenology, fuzzy controllers, reactive baselines, water balance, AquaCrop bridge, and simulation logic.
- `data/cuaca-complete.txt`: weather input.
- `data/paddy_growth_phenology.csv`: paddy phenology table.
- `analysis/` and `scripts/`: comparison, plotting, and sensitivity checks.

## Initial Research Questions

1. Does a phenology-aware fuzzy controller reduce irrigation water use while protecting dynamic soil-moisture target-band tracking?
2. Which part of the contribution is strongest: dynamic phenology target bands, fuzzy inference, or the simulation/reproducibility protocol?
3. Under which seasons or water-stress regimes does the controller help or fail?
4. Are yield-oriented outcomes and IWUE improved without overclaiming beyond simulation evidence?

## Boundaries

- Simulation-only unless field data are later supplied.
- Literature novelty and target-band assumptions require RAG/full-text verification.
- Controller-superiority claims must be based on actual run artifacts, not design intention.
