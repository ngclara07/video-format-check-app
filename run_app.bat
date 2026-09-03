@echo off

title CM3065 - Exercise 3 Video Format Compliance

cd /d "%~dp0"

echo.
echo =====================================================
echo       CM3065 - MEDIA COMPLIANCE LAB
echo       EXERCISE 3 - VIDEO FORMAT CHECK
echo =====================================================
echo.

where ffprobe >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffprobe was not found on PATH.
    echo.
    echo Recommended Conda installation:
    echo conda install -c conda-forge ffmpeg -y
    echo.
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg was not found on PATH.
    echo.
)

python -m streamlit run app.py

pause
