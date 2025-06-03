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
