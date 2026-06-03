# Global Research Context

Last updated: 2026-06-03

## Research Identity

- Project name: `phenology-fuzzy`
- Workspace root: `/home/johanericka/phenology-fuzzy`
- Research mode: structured research-assistant workflow over an existing runnable simulation repository.
- Lead researcher: Johan Erick A.; scientific direction and final choices remain with the lead researcher.
- Assistant role: research coordinator, evidence organizer, experiment planner, simulation/code support, and manuscript support.

## Big Picture

This project studies phenology-aware fuzzy irrigation control for paddy rice. The central idea is that rice soil-moisture targets should be dynamic across growth stages rather than static across the full season. The repository already contains runnable simulation code, BMKG-style weather input data, paddy phenology input data, controller modules, analysis utilities, and reproducibility commands.

## Current Working Scope

- Domain: predictive/smart irrigation, fuzzy control, crop-water simulation.
- Crop: paddy rice.
- Location context: Kabupaten Malang, East Java, Indonesia.
- Weather data: `data/cuaca-complete.txt`.
- Phenology data: `data/paddy_growth_phenology.csv`.
- Main simulation entry point: `main.py`.
- Controller code: `src/fuzzy_controller.py`, `src/fuzzy_static_controller.py`, `src/reactive_controller.py`, `src/reactive_phenology_controller.py`.
- Water-balance/AquaCrop support: `src/water_balance.py`, `src/aquacrop_bridge.py`.
- Analysis utilities: `analysis/compare.py`, `analysis/plots.py`, `scripts/run_sensitivity_checks.py`.

## Tentative Contribution Framing

Tentative, pending literature verification and result audit:

> A phenology-aware fuzzy irrigation controller that uses dynamic stage-specific soil-moisture target bands for paddy rice and evaluates the tradeoff between irrigation water use, target-band tracking, and yield-oriented outcomes under weather-driven closed-loop simulation.

## Evidence Policy

- RAG-first for literature and novelty claims.
- Do not treat existing code comments or prior drafts as literature evidence.
- Full-paper evidence must be stored under `02_literature/references/`, extracted under `02_literature/extractions/`, and tracked in `02_literature/metadata/` before being used for final manuscript claims.
- Simulation claims must be tied to exact commands, configurations, and output artifacts.

## Open Scientific Decisions

1. Confirm the primary paper claim: controller novelty, simulation protocol novelty, or paddy phenology target-band contribution.
2. Confirm target venue type: control/engineering journal, agricultural water management journal, or local/international applied technology journal.
3. Decide whether the existing root-level code remains canonical or should be mirrored into `05_experiments/code/` later.
4. Audit whether existing output artifacts are complete, reproducible, and manuscript-ready.
5. Validate literature support for dynamic phenology-specific soil-moisture targets and fuzzy/MPC irrigation positioning.

## Recommended Next Steps

1. Run an artifact audit of code, data, and generated outputs.
2. Query local RAG for phenology-aware irrigation, paddy water stress, fuzzy irrigation, and AquaCrop simulation evidence.
3. Produce a SOTA/gap memo in `04_gaps/selected-gap.md` or `06_manuscript/sota_presentation.md`.
4. Run or verify the reproducibility command from `README.md` and summarize outputs.
