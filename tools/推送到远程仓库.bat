@echo off
echo === �Զ�ͬ�� Git �ֿ� ===

REM �л��� tools ����Ŀ¼����һ������Ŀ��Ŀ¼��
cd /d "%~dp0.."

REM ��ȡԶ�� main ��֧
echo.
echo --- Pulling from remote repository...
git pull --rebase origin main
IF ERRORLEVEL 1 (
    echo Pull ʧ�ܣ������Ƿ���ڳ�ͻ��
    pause
    exit /b
)

REM ��ӡ��ύ�����͸���
echo.
echo --- Adding and committing changes...
git add .
git commit -m "auto commit"

echo.
echo --- Pushing to remote...
git push origin main
IF ERRORLEVEL 1 (
    echo Push ʧ�ܣ����������Ϣ��
    pause
    exit /b
)

REM �� GitHub �ֿ�ҳ�棨�ɹ���
echo.
echo --- Push �ɹ����� GitHub ҳ��...
start https://github.com/UnbreakablePACHES/spo4portfolio

echo.
echo === ������� ===
pause
