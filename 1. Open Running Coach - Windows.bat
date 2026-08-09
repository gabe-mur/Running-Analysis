@echo off
REM Double-clickable launcher for Windows.
cd /d "%~dp0"

REM The py launcher ships with python.org installs and picks a suitable
REM version; plain `python` on Windows is often a Store stub that does nothing.
where py >nul 2>nul && (py -3 start.py %* & goto :done)
where python >nul 2>nul && (python start.py %* & goto :done)

echo.
echo Running Coach needs Python 3.11 or newer, and none was found.
echo.
echo Install it from https://www.python.org/downloads/ and be sure to tick
echo "Add python.exe to PATH" on the first screen. Then run this file again.
echo.
pause
exit /b 1

:done
if errorlevel 1 pause
