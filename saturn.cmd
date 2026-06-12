@echo off
REM Launch Saturday.ai from anywhere -- no cd, so relative arguments resolve against the
REM CALLER's cwd (and nothing leaks a directory change back into the calling shell).
REM Prefers the repo's own venv interpreter when one exists.
if exist "%~dp0.venv\Scripts\python.exe" ("%~dp0.venv\Scripts\python.exe" "%~dp0agent.py" %*) else (python "%~dp0agent.py" %*)
