@echo off
echo ============================================================
echo HOW LIFESTYLE CHOICES STEAL YOUR YEARS
echo Data Visualization Portfolio Project
echo ============================================================
echo.
echo Activating virtual environment and running project...
echo.

call venv\Scripts\activate.bat
python main.py

echo.
echo ============================================================
echo Project completed! Check the 'exports' folder for outputs.
echo ============================================================
echo.
echo To view the interactive visualization:
echo 1. Run: python -m http.server 8000
echo 2. Open: http://localhost:8000/exports/interactive_visualization.html
echo.
pause

