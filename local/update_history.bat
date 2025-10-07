@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Update history from FINAL results (last 2 days)
if "%SPORTSDATA_API_KEY%"=="" (
  echo ERROR: Please set SPORTSDATA_API_KEY
  exit /b 1
)
if not exist data mkdir data
if not exist models\v3.3 mkdir models\v3.3
if not exist preds mkdir preds
if not exist data\NHL_HISTORY_UNION.csv (
  echo name,team,opponent,home_or_away,date,points> data\NHL_HISTORY_UNION.csv
)
python -m white_shorts.etl.update_history_from_api ^
  --history_csv data/NHL_HISTORY_UNION.csv ^
  --since_days 2 ^
  --key %SPORTSDATA_API_KEY% ^
  --backup
echo Done.
