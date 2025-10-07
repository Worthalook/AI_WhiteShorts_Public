@echo off
setlocal
REM Train models (adjust epochs as desired)
if not exist models\v3.3 mkdir models\v3.3
python train_compare_season.py ^
  --csv data/NHL_HISTORY_UNION.csv ^
  --out_dir models/v3.3 ^
  --split time_per_player ^
  --epochs 2
echo Done.
