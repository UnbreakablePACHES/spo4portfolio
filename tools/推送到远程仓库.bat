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
