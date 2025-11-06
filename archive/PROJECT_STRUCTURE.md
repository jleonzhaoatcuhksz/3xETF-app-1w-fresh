# ETF-App-1w Project Structure

## Core Architecture

```
ETF-App-1w/
├── 🌐 WEB SERVER (Active)
│   └── server_clean.js          # Main Express server (PORT: 3022)
│
├── 📊 DATABASE  
│   └── etf_data.db              # SQLite database with ETF prices
│
├── 🤖 ML STRATEGY ENGINE
│   ├── fast_ml_strategy.py      # Main ML strategy generator
│   └── ml_result_viewer_fixed.py # Results analyzer & viewer
│
├── 📈 ML RESULTS (JSON)
│   ├── random_forest_results.json
│   ├── lightgbm_results.json
│   └── xgboost_results.json
│
└── 📦 DEPENDENCIES
    ├── package.json             # Node.js dependencies
    └── requirements.txt         # Python dependencies
```

## Data Flow

```
1. ETF Data → etf_data.db (SQLite)
2. fast_ml_strategy.py → Generates ML predictions → JSON results
3. server_clean.js → Serves web interface + API endpoints
4. Browser → Displays ETF data, ML results, Trading performance
```

## Key Features

### Web Interface (http://localhost:3022)
- **ETF Data Overview**: Scrollable price data table
- **ML Strategy Results**: Random Forest, XGBoost, LightGBM results
- **Trading Performance**: Detailed trade analysis with modal popups

### API Endpoints
- `/api/etfs` - ETF price data
- `/api/stats` - Database statistics  
- `/api/ml-results` - ML model results
- `/api/trading-performance` - Trading analysis

## File Importance Levels

### 🔴 CRITICAL (Cannot delete)
- server_clean.js
- etf_data.db
- package.json

### 🟡 IMPORTANT (Core functionality)
- fast_ml_strategy.py
- ml_result_viewer_fixed.py
- *_results.json files

### 🟢 OPTIONAL (Can be removed)
- Old server versions (server.js, server_with_ml.js, etc.)
- Debug files (check_db.js, demo.js, showSchema.js)
- Duplicate analysis files