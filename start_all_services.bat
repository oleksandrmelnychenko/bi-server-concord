@echo off
setlocal EnableDelayedExpansion

title BI Server - All Services
color 0A

echo.
echo  ============================================================
echo     BI SERVER - STARTING ALL SERVICES
echo  ============================================================
echo.

:: Configuration
set "PROJECT_ROOT=C:\Users\123\bi-server-concord"
set "AI_BACKEND_PORT=8000"
set "MAIN_API_PORT=8001"
set "DASHBOARD_WS_PORT=8200"
set "FRONTEND_PORT=3000"

:: Kill existing processes
echo  [CLEANUP] Stopping existing services...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo  [CLEANUP] Done.
echo.

:: ============================================================
:: Service 1: AI-Provider Backend (Text-to-SQL)
:: ============================================================
echo  [1/4] Starting AI-Provider Backend (port %AI_BACKEND_PORT%)...
cd /d "%PROJECT_ROOT%\AI-Provider\backend"
start "AI-Provider-Backend-8000" cmd /k "title AI-Provider Backend [%AI_BACKEND_PORT%] && color 0B && python -m uvicorn api:app --host 0.0.0.0 --port %AI_BACKEND_PORT% --reload"
timeout /t 3 /nobreak >nul

:: ============================================================
:: Service 2: Main API (Recommendations, Forecasts, Analytics)
:: ============================================================
echo  [2/4] Starting Main API - Recommendations/Forecasts (port %MAIN_API_PORT%)...
cd /d "%PROJECT_ROOT%\api"
start "Main-API-8001" cmd /k "title Main API [%MAIN_API_PORT%] && color 0D && python -m uvicorn main:app --host 0.0.0.0 --port %MAIN_API_PORT% --reload"
timeout /t 2 /nobreak >nul

:: ============================================================
:: Service 3: Dashboard WebSocket Server
:: ============================================================
echo  [3/4] Starting Dashboard WebSocket Server (port %DASHBOARD_WS_PORT%)...
cd /d "%PROJECT_ROOT%\api"
start "Dashboard-WS-8200" cmd /k "title Dashboard WebSocket [%DASHBOARD_WS_PORT%] && color 0E && python dashboard_server.py"
timeout /t 2 /nobreak >nul

:: ============================================================
:: Service 4: Frontend (React + Vite)
:: ============================================================
echo  [4/4] Starting Frontend (port %FRONTEND_PORT%)...
cd /d "%PROJECT_ROOT%\AI-Provider\frontend"
start "Frontend-3000" cmd /k "title Frontend [%FRONTEND_PORT%] && color 0A && npm run dev"

:: ============================================================
:: Summary
:: ============================================================
echo.
echo  ============================================================
echo     ALL SERVICES STARTED
echo  ============================================================
echo.
echo     Service                         Port      URL
echo     -----------------------------------------------------------
echo     AI-Provider (Text-to-SQL)       %AI_BACKEND_PORT%       http://localhost:%AI_BACKEND_PORT%
echo     Main API (Reco/Forecast)        %MAIN_API_PORT%       http://localhost:%MAIN_API_PORT%
echo     Dashboard WebSocket             %DASHBOARD_WS_PORT%       ws://localhost:%DASHBOARD_WS_PORT%/ws
echo     Frontend                        %FRONTEND_PORT%       http://localhost:%FRONTEND_PORT%
echo.
echo  ============================================================
echo     Open http://localhost:%FRONTEND_PORT% in your browser
echo  ============================================================
echo.
echo     Note: AI-Provider takes ~90 seconds to fully initialize
echo           (loading embedding model and indexing schema)
echo.
echo     Press any key to close this window...
pause >nul
