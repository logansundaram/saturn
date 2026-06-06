@echo off
REM Launch Saturday.ai from anywhere.
cd /d "%~dp0"
python agent.py %*
