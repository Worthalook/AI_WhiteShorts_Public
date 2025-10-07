@echo off
setlocal ENABLEDELAYEDEXPANSION
if "%SPORTSDATA_API_KEY%"=="" (
  echo ERROR: Please set SPORTSDATA_API_KEY
  exit /b 1
)
REM Compute PREDICT_DATE if not provided
if "%PREDICT_DATE%"=="" (
  for /f %%i in ('python "%~dp0..\compute_predict_date.py"') do set PREDICT_DATE=%%i
)
echo Using PREDICT_DATE=%PREDICT_DATE%
python batch_predict_by_player.py ^
  --csv data/NHL_HISTORY_UNION.csv ^
  --head_to_head models/v3.3/per_player_head_to_head.csv ^
  --rf_model models/v3.3/rf_points_forecaster.pkl ^
  --dss_model models/v3.3/dss_model.pt ^
  --date %PREDICT_DATE% ^
  --key %SPORTSDATA_API_KEY% ^
  --out_csv preds\preds_%PREDICT_DATE%.csv ^
  --diagnostics_csv preds\missing_%PREDICT_DATE%.csv
echo Done.
