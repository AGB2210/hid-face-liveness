@echo off
rem Deploy / update / check / remove the AWS version of the kiosk.
rem   deploy.bat            deploy (first time) or show the deployment
rem   deploy.bat status     health check
rem   deploy.bat update     ship current code and restart
rem   deploy.bat destroy    remove everything
setlocal
cd /d "%~dp0"
rem Credentials: aws-credentials.bat if present, otherwise the "hid" profile from "aws login"
if exist "aws-credentials.bat" (
  call "aws-credentials.bat"
) else (
  if not defined AWS_PROFILE set "AWS_PROFILE=hid"
)
if defined AWS_PROFILE echo [deploy] Using AWS profile %AWS_PROFILE% - if it has expired run: aws login --profile %AWS_PROFILE%
if not exist ".venv\Scripts\python.exe" (
  echo [error] Run start.bat once first to set up Python.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\deploy_aws.py %*
pause
