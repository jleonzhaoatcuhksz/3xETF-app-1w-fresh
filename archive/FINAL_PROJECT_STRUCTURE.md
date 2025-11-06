# ETF-App-1w - Simplified Project Structure

## 🎯 FINAL CLEAN ARCHITECTURE

```
ETF-app-1w/
├── 🌐 WEB APPLICATION
│   ├── server_clean.js          # Main Express server (PORT: 3022)
│   ├── package.json             # Node.js dependencies  
│   ├── package-lock.json        # Dependency lock file
│   └── public/                  # Static web files
│
├── 📊 DATABASE & DATA
│   ├── etf_data.db             # Main SQLite database
│   ├── etf_data.db-shm         # SQLite shared memory
│   └── etf_data.db-wal         # SQLite write-ahead log
│
├── 🤖 ML STRATEGY ENGINE
│   ├── fast_ml_strategy.py     # 🔥 MAIN ML strategy generator
│   └── ml_result_viewer_fixed.py # Results analyzer & viewer
│
├── 📈 ML RESULTS (JSON)
│   ├── random_forest_results.json    # Random Forest model results
│   ├── lightgbm_results.json        # LightGBM model results  
│   ├── xgboost_results.json         # XGBoost model results
│   ├── best_fast_ml_results.json    # Best performing model
│   ├── etf_switching_results.json   # ETF switching results
│   └── improved_strategy_results.json # Improved strategy results
│
├── 🧠 ML MODELS (H5)
│   ├── etf_switching_model.h5       # Trained ETF switching model
│   ├── improved_etf_model.h5        # Improved model version
│   └── positive_etf_model.h5        # Positive score model
│
└── 📋 DOCUMENTATION
    ├── PROJECT_STRUCTURE.md         # This file
    ├── README.md                    # Project README
    ├── README_ML_Strategy.md        # ML strategy documentation
    ├── ml_analysis_report.txt       # Analysis report
    └── requirements.txt             # Python dependencies
```

## 🚀 HOW TO USE

### 1. Start the Web Server
```bash
cd ETF-app-1w
node server_clean.js
```
Visit: http://localhost:3022

### 2. Generate New ML Results (Optional)
```bash
python fast_ml_strategy.py
```

### 3. View ML Analysis (Optional)  
```bash
python ml_result_viewer_fixed.py
```

## 🔗 Key Relationships

1. **fast_ml_strategy.py** → Generates → **JSON result files**
2. **server_clean.js** → Reads JSON files → **Web interface**
3. **etf_data.db** → Provides data to → **Both Python & Node.js**

## 📊 Web Interface Features

- **ETF Data Overview**: Scrollable price data table
- **ML Strategy Results**: All 3 models (RF, XGB, LGB) with performance metrics
- **Trading Performance**: Detailed trade analysis with modal popups

## 🎯 SIMPLIFIED FILE COUNT

**Before Cleanup**: 50+ files  
**After Cleanup**: 25 essential files  
**Reduction**: 50% fewer files, 100% functionality retained

## 🔴 CRITICAL FILES (Never Delete)
- server_clean.js
- etf_data.db  
- fast_ml_strategy.py
- package.json