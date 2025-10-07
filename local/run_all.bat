@echo off
setlocal ENABLEDELAYEDEXPANSION
REM One-shot: update → train → predict
call "%~dp0update_history.bat" || exit /b 1
call "%~dp0train.bat" || exit /b 1
call "%~dp0predict.bat" || exit /b 1
echo All steps completed.
