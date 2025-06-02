@echo off
git add .
git commit -m "auto commit"
git push origin main

IF %ERRORLEVEL% EQU 0 (
    echo Push succeeded. Opening GitHub...
    start https://github.com/UnbreakablePACHES/Dlinear-for-Portofolio-optimzation
) ELSE (
    echo Push failed.
)

pause
