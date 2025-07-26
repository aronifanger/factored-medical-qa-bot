# This script starts the backend API and the frontend web app in separate PowerShell windows.
# It should be run from the root of the project directory.

# Get the directory where the script is located, which should be the project root.
$scriptPath = $PSScriptRoot

Write-Host "Project Path: $scriptPath"
Write-Host "---"

# --- Start FastAPI Backend ---
Write-Host "Starting FastAPI server (API)..." -ForegroundColor Green
# The command changes directory to the script's location and starts the uvicorn server.
$fastAPICommand = "cd '$scriptPath'; uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $fastAPICommand

Write-Host "API will be running at http://127.0.0.1:8000"
Write-Host "Waiting for 5 seconds for the API to initialize..."
Start-Sleep -Seconds 5 # Give the API a moment to start to avoid frontend errors on startup.

# --- Start Streamlit Frontend ---
Write-Host "---"
Write-Host "Starting Streamlit application (Frontend)..." -ForegroundColor Green
# The command changes directory and starts the streamlit app.
$streamlitCommand = "cd '$scriptPath'; streamlit run app/App.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $streamlitCommand

Write-Host "Frontend will be running at http://localhost:8501 (a browser tab should open automatically)."
Write-Host "---"
Write-Host "All services have been launched successfully in new windows." -ForegroundColor Cyan 