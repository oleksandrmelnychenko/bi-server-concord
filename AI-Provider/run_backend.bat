@echo off
cd /d "%~dp0backend"
echo Starting AI-Provider Backend on http://localhost:8000
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
