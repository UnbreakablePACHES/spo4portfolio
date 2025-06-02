@echo off
chcp 65001 >nul
echo === 自动拉取 Git 仓库 ===

echo.
echo --- Pulling from remote repository...
git pull

if %errorlevel% neq 0 (
    echo.
    echo !!! 拉取失败，请检查是否有未提交的更改或网络问题 !!!
) else (
    echo.
    echo --- 拉取成功，工作区已更新 ---
)

echo.
echo === 操作完成 ===
pause
