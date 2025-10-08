@echo off
setlocal ENABLEDELAYEDEXPANSION

if "%SPORTSDATA_API_KEY%"=="" (
  echo [update_history] ERROR: Please set SPORTSDATA_API_KEY
  goto :EOF
)

if not exist data mkdir data
if not exist models\v3.3 mkdir models\v3.3
if not exist preds mkdir preds
if not exist data\NHL_HISTORY_UNION.csv (
  echo name,team,opponent,home_or_away,date,points> data\NHL_HISTORY_UNION.csv
)

python -m white_shorts.etl.update_history_from_api --history_csv data/NHL_HISTORY_UNION.csv --since_days 2 --key %SPORTSDATA_API_KEY% --backup
if %ERRORLEVEL% NEQ 0 (
  echo [update_history] WARNING: Update history not available yet (timeout or API error). Continuing...
) else (
  echo [update_history] OK: history refreshed.
)
exit /b 0
