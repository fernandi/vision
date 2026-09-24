@echo off
rem Glane : version locale du front, branchee sur l'API de production.
cd /d "%~dp0"
start "" http://localhost:5173
set PYTHONIOENCODING=utf-8
python dev_server.py
pause
