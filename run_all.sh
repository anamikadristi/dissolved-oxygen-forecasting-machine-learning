#!/usr/bin/env bash
# Run the full pipeline from download to figures.
# Usage: bash run_all.sh [raw_data_dir]
set -euo pipefail
RAW="${1:-raw_data}"
cd "$(dirname "$0")"
python scripts/01_download_data.py --raw-dir "$RAW"
python scripts/02_build_daily_dataset.py --raw-dir "$RAW"
python scripts/03_train_lstm.py
python scripts/04_train_forecast_models.py
python scripts/05_hypoxia_classification.py
python scripts/06_shap_interpretability.py
python scripts/07_cross_site_transfer.py
python scripts/08_make_figures_tables.py
