@echo off
setlocal
if not exist models\v3.3 mkdir models\v3.3
python train_compare_season.py ^
  --csv_ytd data/NHL_YTD.csv ^
  --csv_last data/NHL_2023_24.csv ^
  --out_dir models\v3.3 ^
  --split time_per_player ^
  --epochs 2
if errorlevel 1 (
  echo [train] WARNING: Training encountered an error. Continuing.
) else (
  echo [train] OK: models updated.
)
