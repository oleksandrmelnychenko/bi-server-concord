@echo off
cd /d "%~dp0frontend"
echo Starting AI-Provider Frontend on http://localhost:3000
call npm install
call npm run dev -- --host 0.0.0.0 --port 3000
