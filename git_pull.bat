@echo off
echo Pulling from remote repository...

REM 拉取远程 main 分支
git pull --rebase origin main

IF %ERRORLEVEL% EQU 0 (
    echo Pull succeeded.
) ELSE (
    echo Pull failed. Please check for conflicts or remote changes.
)

pause