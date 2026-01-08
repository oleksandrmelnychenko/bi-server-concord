@echo off
title BI Server - Stop All Services
color 0C

echo.
echo  ============================================================
echo     BI SERVER - STOPPING ALL SERVICES
echo  ============================================================
echo.

echo  [1/2] Stopping Python services (API backends)...
taskkill /F /IM python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo        Python processes stopped.
) else (
    echo        No Python processes found.
)

echo  [2/2] Stopping Node services (Frontend)...
taskkill /F /IM node.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo        Node processes stopped.
) else (
    echo        No Node processes found.
)

echo.
echo  ============================================================
echo     ALL SERVICES STOPPED
echo  ============================================================
echo.
echo     Press any key to close...
pause >nul
