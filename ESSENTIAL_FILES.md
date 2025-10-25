# Essential Files for ETF Dashboard System

This file lists all essential files required for running the ETF dashboard system with ML strategy analysis.

## Core Server Files

### Price Search Server (Port 8086)
- `price_search_server.py` - Main server file
- `price_search.html` - Web interface
- `etf_data.db` - Database with ETF price data

### Test Dashboard Server (Port 8080)  
- `start_test_dashboard_server.py` - Main server file
- `ml_strategy_dashboard_test_only.html` - Dashboard interface
- `single_etf_ml_switching_results_test_only.json` - Test period results

## ML Model Files (Required for Dashboard)

### Model Files
- `etf_switching_model.h5` - Main ML model
- `improved_etf_model.h5` - Improved ML model  
- `positive_etf_model.h5` - Positive ETF model

### Results Files
- `single_etf_ml_switching_results.json` - Full ML results
- `xgboost_results.json` - XGBoost results
- `lightgbm_results.json` - LightGBM results
- `random_forest_results.json` - Random Forest results
- `improved_strategy_results.json` - Improved strategy
- `best_fast_ml_results.json` - Fast ML results
- `etf_switching_results.json` - ETF switching results

### Supporting Files
- `monthly_returns_data.json` - Monthly returns data
- `ml_analysis_report.txt` - ML analysis report

## Other Essential Files
- `server_clean.js` - Alternative Node.js server
- `requirements.txt` - Python dependencies
- `package.json` - Node.js dependencies

## Backup Files
All other Python scripts have been moved to the `backup/` folder.

## Server Commands
```bash
# Start test dashboard server
python start_test_dashboard_server.py

# Start price search server  
python price_search_server.py

# URLs
Dashboard: http://localhost:8080
Price Search: http://localhost:8086
```

## Test Period
The dashboard displays data for the testing period: **2024-08-20 to 2025-10-17**

Created: 2025-10-24