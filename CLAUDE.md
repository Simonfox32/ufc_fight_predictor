# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UFC Fight Predictor is an end-to-end ML project that predicts UFC fight outcomes. It scrapes data from ufcstats.com, engineers features with strict no-leakage rules, and trains classification models.

**Stack:** Python 3, pandas, scikit-learn, XGBoost, requests, BeautifulSoup4

## Commands

```bash
# Run full scraping pipeline (01-06 scripts sequentially)
python src/scraping/07_make_dataset.py

# Feature engineering (v2a → v2b)
python src/features/build_features.py

# Train models
python src/models/logistic_regression.py
python src/models/xgboost_model.py

# Run evaluation notebook (uses # %% cell markers)
python model_eval.py

# Environment setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Architecture

```
Data Pipeline:
src/scraping/01_events.py          → data_raw/extracts/all_events.csv
src/scraping/02_fights_from_events.py → data_raw/extracts/all_fights.csv
src/scraping/03_data_from_fights.py   → data_raw/extracts/rounds_parsed.csv
src/scraping/04_reshape_rounds_to_fighters.py → round_stats_long.csv
src/scraping/05_fight_total_aggregate.py → fight_agg.csv
src/scraping/06_fighter_list.py       → fighters_stats.csv

Feature Engineering:
src/features/build_features.py    → data_processed/model_data/fight_model_v2b.csv

Models:
src/models/logistic_regression.py → models/ufc_lr_pipeline_*.joblib
src/models/xgboost_model.py       → models/ufc_xgb_*.joblib
```

## Critical: No-Leakage Rules

All features must use only pre-fight data. See [features/FEATURE_SPEC.md](features/FEATURE_SPEC.md) for complete spec.

- **As-of rule:** Only use fights where `prior_fight_date < fight_date`
- **Eligibility:** Both fighters must have ≥5 prior fights
- **L5 window:** Rolling features use `shift(1).rolling(5, min_periods=1)` to prevent leakage
- **Train/test split:** Time-based at 2022-01-01

## Model Output Format

Models are saved as joblib bundles:
```python
{
    "model": pipeline_or_classifier,
    "feature_cols": list,
    "cutoff_date": str,
    "train_rows": int,
    "test_rows": int
}
```

## Key Feature Categories

- **Form (L5):** win rate, avg fight duration, days since last fight
- **Striking (L5):** sig strikes landed/min, accuracy, knockdowns/min
- **Grappling (L5):** control time/15min, takedowns/15min, TD accuracy
- **Static:** age, height, reach, stance (southpaw flags)
- **Deltas:** red - blue differences for all metrics

## Data Paths

- Raw HTML: `data_raw/html/`
- Extracted CSVs: `data_raw/extracts/`
- Model-ready data: `data_processed/model_data/fight_model_v2b.csv`
- Trained models: `models/*.joblib`
