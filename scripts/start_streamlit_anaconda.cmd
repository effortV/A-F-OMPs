@echo off
setlocal
cd /d "%~dp0.."
set "API_BASE_URL=http://localhost:8000"
where conda >nul 2>nul
if errorlevel 1 (
  echo Conda was not found. Open this script from Anaconda Prompt.
  pause
  exit /b 1
)
echo Starting NF-Atlas Streamlit with Conda environment: nf-agent
conda run --no-capture-output -n nf-agent streamlit run ui\streamlit_app.py --server.port 8501
endlocal

