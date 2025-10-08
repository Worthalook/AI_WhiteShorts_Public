@echo off
setlocal ENABLEDELAYEDEXPANSION
call "%~dp0update_history.bat"
call "%~dp0train.bat"
call "%~dp0predict.bat"
echo [run_all] Completed.
