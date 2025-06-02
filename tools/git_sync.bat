@echo off
echo === 自动同步 Git 仓库 ===

REM 切换到 tools 所在目录的上一级（项目根目录）
cd /d "%~dp0.."

REM 拉取远程 main 分支
echo.
echo --- Pulling from remote repository...
git pull --rebase origin main
IF ERRORLEVEL 1 (
    echo Pull 失败，请检查是否存在冲突。
    pause
    exit /b
)

REM 添加、提交并推送更改
echo.
echo --- Adding and committing changes...
git add .
git commit -m "auto commit"

echo.
echo --- Pushing to remote...
git push origin main
IF ERRORLEVEL 1 (
    echo Push 失败，请检查错误信息。
    pause
    exit /b
)

REM 打开 GitHub 仓库页面（成功后）
echo.
echo --- Push 成功，打开 GitHub 页面...
start https://github.com/UnbreakablePACHES/spo4portfolio

echo.
echo === 操作完成 ===
pause
