@echo off
setlocal ENABLEDELAYEDEXPANSION
if "%SPORTSDATA_API_KEY%"=="" (
  echo [predict] ERROR: Please set SPORTSDATA_API_KEY
  goto :EOF
)
if "%PREDICT_DATE%"=="" (
  for /f %%i in ('python "%~dp0..\compute_predict_date.py"') do set PREDICT_DATE=%%i
)
echo [predict] Using PREDICT_DATE=%PREDICT_DATE%
for /f "usebackq delims=" %%C in (`python -c "import sys,urllib.request,json; import ssl; ssl._create_default_https_context=ssl._create_unverified_context; import os; key=os.environ.get('SPORTSDATA_API_KEY'); import urllib.parse as up; from datetime import datetime; d=os.environ.get('PREDICT_DATE'); mon=datetime.strptime(d,'%Y-%m-%d').strftime('%Y-%%b-%d'); url=f'https://api.sportsdata.io/api/nhl/fantasy/json/PlayerGameProjectionStatsByDate/{mon}?key={key}';\ntry:\n  with urllib.request.urlopen(url,timeout=30) as r:\n    data=json.loads(r.read().decode('utf-8'))\n  print(len(data) if isinstance(data,list) else 0)\nexcept Exception as e:\n  print('ERR')"` ) do set COUNT=%%C
if /I "%COUNT%"=="ERR" (
  echo [predict] WARNING: Projections API unreachable or timed out. Skipping prediction this run.
  goto :EOF
)
set /a CNTNUM=%COUNT% 2>nul
if "%CNTNUM%"=="" set CNTNUM=0
if %CNTNUM% LSS 10 (
  echo [predict] INFO: Projections not ready yet (count=%CNTNUM%). Skipping prediction.
  goto :EOF
)
echo [predict] Projections ready (count=%CNTNUM%). Running batch prediction...
python batch_predict_by_player.py ^
  --csv data/NHL_HISTORY_UNION.csv ^
  --head_to_head models\v3.3\per_player_head_to_head.csv ^
  --rf_model models\v3.3\rf_points_forecaster.pkl ^
  --dss_model models\v3.3\dss_model.pt ^
  --date %PREDICT_DATE% ^
  --key %SPORTSDATA_API_KEY% ^
  --out_csv preds\preds_%PREDICT_DATE%.csv ^
  --diagnostics_csv preds\missing_%PREDICT_DATE%.csv
if errorlevel 1 (
  echo [predict] ERROR: batch prediction failed.
) else (
  echo [predict] OK: predictions written to preds\preds_%PREDICT_DATE%.csv
)
