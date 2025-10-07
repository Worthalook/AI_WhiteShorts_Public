# ============================================================================
#  AI WhiteShorts — GP_Refactor Branch
#  Nightly automation pipeline: update → train → predict
#  Usage:
#     make update     # merge FINAL game results into union history CSV
#     make train      # retrain RF + DSS models (weighted YTD + Last)
#     make predict    # run projections forecast for the next game date
# ============================================================================

# ----- CONFIGURATION --------------------------------------------------------
PY=python
DATA_DIR?=data
MODELS_DIR?=models/v3.3
PREDS_DIR?=preds

HIST_CSV?=$(DATA_DIR)/NHL_HISTORY_UNION.csv
YTD_CSV?=$(DATA_DIR)/NHL_YTD.csv
LAST_CSV?=$(DATA_DIR)/NHL_2023_24.csv

# Environment variables:
#   SPORTSDATA_API_KEY  → required for update & predict
#   PREDICT_DATE        → required for predict (e.g. 2025-05-07 or 2025-May-07)
# Example:
#   export SPORTSDATA_API_KEY=xxxxxx
#   export PREDICT_DATE=2025-05-07

# ----- TARGETS --------------------------------------------------------------
.PHONY: dirs update train predict

dirs:
	@mkdir -p $(DATA_DIR) $(MODELS_DIR) $(PREDS_DIR)

# Update union history CSV with FINAL results
update: dirs
	@echo "== [UPDATE] Merging FINAL game results into $(HIST_CSV) =="
	@if [ -z "$$SPORTSDATA_API_KEY" ]; then echo "ERROR: Set SPORTSDATA_API_KEY"; exit 1; fi
	$(PY) -m white_shorts.etl.update_history_from_api \
	  --history_csv $(HIST_CSV) \
	  --since_days 2 \
	  --key $$SPORTSDATA_API_KEY \
	  --backup

# Train both RandomForest & DSS models (weighted YTD + last season)
train: dirs
	@echo "== [TRAIN] Writing artifacts to $(MODELS_DIR) =="
	$(PY) train_compare_season.py \
	  --csv_ytd $(YTD_CSV) \
	  --csv_last $(LAST_CSV) \
	  --last_weight 0.5 \
	  --out_dir $(MODELS_DIR) \
	  --split time_per_player \
	  --epochs 8

# Predict next-game points using projections API
predict: dirs
	@echo "== [PREDICT] Running forecast for $(PREDICT_DATE) =="
	@if [ -z "$$SPORTSDATA_API_KEY" ]; then echo "ERROR: Set SPORTSDATA_API_KEY"; exit 1; fi
	@if [ -z "$$PREDICT_DATE" ]; then echo "ERROR: Set PREDICT_DATE (e.g. 2025-05-07)"; exit 1; fi
	$(PY) batch_predict_by_player.py \
	  --csv $(HIST_CSV) \
	  --head_to_head $(MODELS_DIR)/per_player_head_to_head.csv \
	  --rf_model $(MODELS_DIR)/rf_points_forecaster.pkl \
	  --dss_model $(MODELS_DIR)/dss_model.pt \
	  --date $$PREDICT_DATE \
	  --key $$SPORTSDATA_API_KEY \
	  --out_csv $(PREDS_DIR)/preds_$$(echo $$PREDICT_DATE | tr '/' '_' | tr ' ' '_').csv \
	  --diagnostics_csv $(PREDS_DIR)/missing_$$(echo $$PREDICT_DATE | tr '/' '_' | tr ' ' '_').csv

# Clean intermediate artifacts (optional)
clean:
	@echo "Cleaning temp files..."
	rm -rf $(MODELS_DIR)/*.tmp $(PREDS_DIR)/*.tmp
