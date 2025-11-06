@echo off
echo Stopping existing servers...

taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1

timeout /t 2 /nobreak >nul

echo Starting servers...

start "ETF Dashboard" powershell -NoExit -Command "cd 'c:\Users\JLZ\codebuddy\Projects\3xETF-app-1w-lean-fresh'; node server_clean.js"
timeout /t 2 /nobreak >nul

start "Price Search" powershell -NoExit -Command "cd 'c:\Users\JLZ\codebuddy\Projects\3xETF-app-1w-lean-fresh'; python price_search_server.py"
timeout /t 2 /nobreak >nul

start "Data Browser" powershell -NoExit -Command "cd 'c:\Users\JLZ\codebuddy\Projects\3xETF-app-1w-lean-fresh'; python data_browser_server.py"
timeout /t 2 /nobreak >nul

start "ML Dashboard" powershell -NoExit -Command "cd 'c:\Users\JLZ\codebuddy\Projects\3xETF-app-1w-lean-fresh'; python start_dashboard_server.py"

echo All servers started!
echo.
echo Access URLs:
echo Main Dashboard: http://localhost:3022
echo ML Strategy: http://localhost:8080/ml_strategy_dashboard.html
echo Price Search: http://localhost:8086/price_search.html
echo Data Browser: http://localhost:8087/data_browser.html
