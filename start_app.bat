@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env. Add your GEMINI_API_KEY before using LLM answers.
    notepad .env
)
streamlit run app.py
if errorlevel 1 goto :error
goto :eof
:error
echo.
echo Setup or launch failed. Review the error above.
pause
