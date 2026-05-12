# Phenology-Fuzzy Irrigation Simulation

This repository contains the runnable simulation code and input data needed to reproduce the phenology-aware fuzzy irrigation controller study. 

## Contents

- `main.py`: main simulation entry point.
- `src/`: controller, phenology, water-balance, AquaCrop bridge, and simulation modules.
- `analysis/`: comparison statistics and plotting utilities.
- `data/`: input weather and paddy phenology tables used by the simulation.
- `scripts/run_sensitivity_checks.py`: robustness sensitivity checks.
- `output/reproducibility_2015_2024/`: generated CSV/JSON/TXT/PNG artifacts from reproducibility runs. This directory is ignored and can be regenerated.

## Setup

Linux/macOS:

```bash
./bootstrap.sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Reproduce Main Evaluation

```bash
python main.py --reproducibility-pack \
  --reproducibility-start-year 2015 \
  --reproducibility-end-year 2024 \
  --experiment-mode all
```

Primary outputs are written under:

```text
output/reproducibility_2015_2024/
```

## Reproduce Sensitivity Checks

```bash
python scripts/run_sensitivity_checks.py --with-aquacrop
```

The sensitivity outputs are written under:

```text
output/reproducibility_2015_2024/sensitivity/
```

## Data

The public data files are:

- `data/cuaca-complete.txt`
- `data/paddy_growth_phenology.csv`