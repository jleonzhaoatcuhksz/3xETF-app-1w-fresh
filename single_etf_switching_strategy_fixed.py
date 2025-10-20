#!/usr/bin/env python3
"""
Single ETF Switching Strategy with ML Algorithms
Uses XGBoost, LightGBM, and Random Forest to determine optimal ETF switching
Based on Monthly_Trend analysis for 3X leveraged ETFs
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb

class SingleETFSwitchingStrategy:
    """
    ML-based single ETF switching strategy that determines:
    1. Which ETF to switch to (target selection)
    2. When to switch (timing optimization)
    3. Only switches when profitable based on Monthly_Trend predictions
    """
    
    def __init__(self, db_path='etf_data.db'):
        self.db_path = db_path
        self.data = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.results = {}
        
    def load_data(self):
        """Load and prepare data from SQLite database"""
        print("📊 Loading 3X ETF data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load price data with additional ETF information
        query = """
        SELECT p.*, e.name, e.sector, e.category
        FROM prices p
        LEFT JOIN etfs e ON p.symbol = e.symbol
        ORDER BY p.symbol, p.date
        """
        
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(self.data)} records for {self.data['symbol'].nunique()} ETFs")
        print(f"📅 Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        # Convert date column
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        return self.data
    
    def create_features(self):
        """Create comprehensive feature set for ML models"""
        print("🔧 Creating features for ML models...")
        
        # Fill missing values in critical columns
        self.data['close'] = self.data['close'].ffill()
        self.data['monthly_trend'] = self.data['monthly_trend'].fillna(0)
        self.data['sma_5d'] = self.data['sma_5d'].ffill()
        
        # Sort data by symbol and date
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate additional technical indicators
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # Price-based features
            symbol_data['returns_1d'] = symbol_data['close'].pct_change().fillna(0)
            symbol_data['returns_5d'] = symbol_data['close'].pct_change(periods=5).fillna(0)
            symbol_data['returns_10d'] = symbol_data['close'].pct_change(periods=10).fillna(0)
            
            # Volatility features  
            symbol_data['volatility_5d'] = symbol_data['returns_1d'].rolling(5).std().fillna(0)
            symbol_data['volatility_10d'] = symbol_data['returns_1d'].rolling(10).std().fillna(0)
            
            # Moving averages
            symbol_data['sma_10d'] = symbol_data['close'].rolling(10).mean().ffill()
            symbol_data['sma_20d'] = symbol_data['close'].rolling(20).mean().ffill()
            
            # Relative strength
            symbol_data['rsi_14'] = self.calculate_rsi(symbol_data['close'], 14).fillna(50)
            
            # Price momentum
            symbol_data['momentum_5d'] = (symbol_data['close'] / symbol_data['close'].shift(5) - 1).fillna(0)
            symbol_data['momentum_10d'] = (symbol_data['close'] / symbol_data['close'].shift(10) - 1).fillna(0)
            
            # Update main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        # Create cross-sectional features (compare with other ETFs)
        self.data = self.create_cross_sectional_features()
        
        # Create target variables
        self.data = self.create_targets()
        
        # Select feature columns
        self.feature_columns = [
            'close', 'sma_5d', 'monthly_trend', 'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d', 'sma_10d', 'sma_20d', 'rsi_14',
            'momentum_5d', 'momentum_10d', 'relative_strength', 'rank_monthly_trend',
            'sector_encoded', 'category_encoded'
        ]
        
        print(f"✅ Created {len(self.feature_columns)} features")
        return self.data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_cross_sectional_features(self):
        """Create features comparing ETFs at each date"""
        print("🔄 Creating cross-sectional features...")
        
        # Encode categorical variables
        le_sector = LabelEncoder()
        le_category = LabelEncoder()
        
        self.data['sector_encoded'] = le_sector.fit_transform(self.data['sector'].fillna('Unknown'))
        self.data['category_encoded'] = le_category.fit_transform(self.data['category'].fillna('Unknown'))
        
        self.label_encoders['sector'] = le_sector
        self.label_encoders['category'] = le_category
        
        # For each date, calculate relative performance
        date_groups = self.data.groupby('date')
        
        relative_features = []
        for date, group in date_groups:
            group = group.copy()
            
            # Rank ETFs by monthly trend
            group['rank_monthly_trend'] = group['monthly_trend'].rank(ascending=False)
            
            # Relative strength vs average
            avg_monthly_trend = group['monthly_trend'].mean()
            group['relative_strength'] = group['monthly_trend'] - avg_monthly_trend
            
            relative_features.append(group)
        
        result_df = pd.concat(relative_features).sort_values(['symbol', 'date'])
        return result_df
    
    def create_targets(self):
        """Create target variables for ML models"""
        print("🎯 Creating target variables...")
        
        # Sort by date to ensure proper forward-looking targets
        self.data = self.data.sort_values('date')
        
        # For each date, identify the best ETF to switch to
        date_groups = self.data.groupby('date')
        
        target_data = []
        for date, group in date_groups:
            group = group.copy()
            
            # Handle missing values in monthly_trend
            valid_trends = group['monthly_trend'].dropna()
            if len(valid_trends) == 0:
                # If no valid trends, skip this date
                continue
            
            # Find ETF with highest monthly trend (best performer)
            best_idx = group['monthly_trend'].idxmax()
            if pd.isna(best_idx):
                # Skip if no valid maximum found
                continue
            
            best_etf = group.loc[best_idx, 'symbol']
            
            # Create binary target: 1 if this ETF is the best choice, 0 otherwise
            group['is_best_choice'] = (group['symbol'] == best_etf).astype(int)
            
            # Create switching signal: 1 if monthly trend > threshold, 0 otherwise
            valid_trends = group['monthly_trend'].dropna()
            if len(valid_trends) > 0:
                threshold = valid_trends.quantile(0.75)  # Top 25%
                group['should_switch'] = (group['monthly_trend'] > threshold).fillna(0).astype(int)
            else:
                group['should_switch'] = 0
            
            # Multi-class target: which ETF to choose (label encoding)
            group['target_etf'] = group['symbol']
            
            target_data.append(group)
        
        if len(target_data) == 0:  
            raise ValueError("No valid target data created. Check for missing values in monthly_trend.")
        
        result_df = pd.concat(target_data).sort_values(['symbol', 'date'])
        
        # Encode target ETF labels
        le_target = LabelEncoder()
        result_df['target_etf_encoded'] = le_target.fit_transform(result_df['target_etf'])
        self.label_encoders['target_etf'] = le_target
        
        return result_df
    
    def prepare_ml_data(self):
        """Prepare data for ML training"""
        print("🔄 Preparing ML training data...")
        
        # Remove rows with missing values
        ml_data = self.data[self.feature_columns + ['is_best_choice', 'should_switch', 'target_etf_encoded']].dropna()
        
        X = ml_data[self.feature_columns]
        y_best = ml_data['is_best_choice']  # Binary: is this the best ETF?
        y_switch = ml_data['should_switch']  # Binary: should we switch?
        y_target = ml_data['target_etf_encoded']  # Multi-class: which ETF to choose?
        
        print(f"✅ Prepared {len(X)} samples with {len(self.feature_columns)} features")
        print(f"📊 Target distribution - Best choice: {y_best.value_counts().to_dict()}")
        print(f"📊 Switch signal distribution: {y_switch.value_counts().to_dict()}")
        
        return X, y_best, y_switch, y_target
    
    def train_models(self):
        """Train XGBoost, LightGBM, and Random Forest models"""
        print("🤖 Training ML models...")
        
        X, y_best, y_switch, y_target = self.prepare_ml_data()
        
        # Split data chronologically (use earlier data for training)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_best_train, y_best_test = y_best.iloc[:split_idx], y_best.iloc[split_idx:]
        y_switch_train, y_switch_test = y_switch.iloc[:split_idx], y_switch.iloc[split_idx:]
        y_target_train, y_target_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Train models for best ETF selection (binary classification)
        print("\n🎯 Training Best ETF Selection Models...")
        
        # XGBoost
        xgb_best = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_best.fit(X_train_scaled, y_best_train)
        
        # LightGBM
        lgb_best = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_best.fit(X_train_scaled, y_best_train)
        
        # Random Forest
        rf_best = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_best.fit(X_train_scaled, y_best_train)
        
        # Store models
        self.models['xgb_best'] = xgb_best
        self.models['lgb_best'] = lgb_best
        self.models['rf_best'] = rf_best
        
        # Train models for switching timing (binary classification)
        print("\n⏰ Training Switch Timing Models...")
        
        xgb_switch = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_switch.fit(X_train_scaled, y_switch_train)
        
        lgb_switch = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_switch.fit(X_train_scaled, y_switch_train)
        
        rf_switch = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_switch.fit(X_train_scaled, y_switch_train)
        
        self.models['xgb_switch'] = xgb_switch
        self.models['lgb_switch'] = lgb_switch
        self.models['rf_switch'] = rf_switch
        
        # Train models for target ETF selection (multi-class)
        print("\n🎲 Training Target ETF Selection Models...")
        
        xgb_target = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_target.fit(X_train_scaled, y_target_train)
        
        lgb_target = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_target.fit(X_train_scaled, y_target_train)
        
        rf_target = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_target.fit(X_train_scaled, y_target_train)
        
        self.models['xgb_target'] = xgb_target
        self.models['lgb_target'] = lgb_target
        self.models['rf_target'] = rf_target
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_best_test, y_switch_test, y_target_test)
        
        print("✅ All models trained successfully!")
    
    def evaluate_models(self, X_test, y_best_test, y_switch_test, y_target_test):
        """Evaluate model performance"""
        print("\n📈 Evaluating Model Performance...")
        
        model_performance = {}
        
        # Evaluate best ETF selection models
        for model_name in ['xgb_best', 'lgb_best', 'rf_best']:
            pred = self.models[model_name].predict(X_test)
            accuracy = accuracy_score(y_best_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Best ETF Accuracy: {accuracy:.4f}")
        
        # Evaluate switch timing models
        for model_name in ['xgb_switch', 'lgb_switch', 'rf_switch']:
            pred = self.models[model_name].predict(X_test)
            accuracy = accuracy_score(y_switch_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Switch Timing Accuracy: {accuracy:.4f}")
        
        # Evaluate target ETF selection models
        for model_name in ['xgb_target', 'lgb_target', 'rf_target']:
            pred = self.models[model_name].predict(X_test)
            accuracy = accuracy_score(y_target_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Target ETF Accuracy: {accuracy:.4f}")
        
        self.results['model_performance'] = model_performance
    
    def run_backtest(self, start_date='2020-01-01', initial_capital=10000):
        """Run comprehensive backtest using ensemble of ML models"""
        print(f"\n🚀 Running backtest from {start_date} with ${initial_capital:,} initial capital...")
        
        # Filter data for backtest period
        backtest_data = self.data[self.data['date'] >= start_date].copy()
        backtest_data = backtest_data.sort_values('date')
        
        # Initialize portfolio
        portfolio_value = initial_capital
        current_etf = None
        trades = []
        portfolio_history = []
        
        # Get unique dates for iteration
        dates = sorted(backtest_data['date'].unique())
        
        for i, date in enumerate(dates[50:]):  # Skip first 50 days for feature calculation
            
            # Get data for current date
            current_date_data = backtest_data[backtest_data['date'] == date]
            
            if len(current_date_data) == 0:
                continue
            
            # Prepare features for prediction
            features_for_prediction = []
            for _, row in current_date_data.iterrows():
                if not pd.isna(row[self.feature_columns]).any():
                    features_for_prediction.append(row[self.feature_columns].values)
            
            if len(features_for_prediction) == 0:
                continue
            
            X_current = np.array(features_for_prediction)
            X_current_scaled = self.scalers['main'].transform(X_current)
            
            # Get ensemble predictions for each ETF
            ensemble_scores = []
            etf_symbols = current_date_data['symbol'].tolist()
            
            for j, (_, row) in enumerate(current_date_data.iterrows()):
                if j >= len(X_current_scaled):
                    break
                
                # Get predictions from all models
                best_scores = []
                switch_scores = []
                
                # Best ETF predictions
                for model_name in ['xgb_best', 'lgb_best', 'rf_best']:
                    pred_proba = self.models[model_name].predict_proba(X_current_scaled[j:j+1])[0]
                    best_scores.append(pred_proba[1])  # Probability of being best
                
                # Switch timing predictions
                for model_name in ['xgb_switch', 'lgb_switch', 'rf_switch']:
                    pred_proba = self.models[model_name].predict_proba(X_current_scaled[j:j+1])[0]
                    switch_scores.append(pred_proba[1])  # Probability of switching
                
                # Ensemble score (average of all predictions)
                best_score = np.mean(best_scores)
                switch_score = np.mean(switch_scores)
                combined_score = (best_score * 0.6 + switch_score * 0.4)  # Weight best ETF more
                
                ensemble_scores.append({
                    'symbol': row['symbol'],
                    'best_score': best_score,
                    'switch_score': switch_score,
                    'combined_score': combined_score,
                    'monthly_trend': row['monthly_trend'],
                    'close': row['close']
                })
            
            if len(ensemble_scores) == 0:
                continue
            
            # Find best ETF according to ensemble
            best_etf_data = max(ensemble_scores, key=lambda x: x['combined_score'])
            best_etf = best_etf_data['symbol']
            
            # Decision logic: Switch if profitable and confidence is high
            should_switch = False
            
            if current_etf is None:
                # Initial investment
                should_switch = True
            else:
                # Check if switching is profitable
                current_etf_data = next((x for x in ensemble_scores if x['symbol'] == current_etf), None)
                
                if current_etf_data and best_etf_data:
                    # Switch if:
                    # 1. Best ETF has significantly higher score
                    # 2. Best ETF has positive monthly trend
                    # 3. Current ETF is underperforming
                    score_improvement = best_etf_data['combined_score'] - current_etf_data['combined_score']
                    
                    if (score_improvement > 0.1 and  # Significant improvement
                        best_etf_data['monthly_trend'] > 0.02 and  # Positive trend
                        best_etf_data['combined_score'] > 0.6):  # High confidence
                        should_switch = True
            
            # Execute trade if switching
            if should_switch and best_etf != current_etf:
                
                # Calculate portfolio value change if we had previous position
                if current_etf is not None:
                    # Find previous price
                    prev_date_idx = max(0, i - 1)
                    prev_date = dates[prev_date_idx + 50]  # Adjust for skip
                    prev_data = backtest_data[
                        (backtest_data['date'] == prev_date) & 
                        (backtest_data['symbol'] == current_etf)
                    ]
                    
                    if len(prev_data) > 0:
                        prev_price = prev_data.iloc[0]['close']
                        current_price = current_etf_data['close'] if current_etf_data else prev_price
                        
                        # Update portfolio value based on previous position
                        price_change = (current_price - prev_price) / prev_price
                        portfolio_value *= (1 + price_change)
                
                # Record trade
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'from_etf': current_etf,
                    'to_etf': best_etf,
                    'confidence': best_etf_data['combined_score'],
                    'monthly_trend': best_etf_data['monthly_trend'],
                    'portfolio_value': portfolio_value
                })
                
                current_etf = best_etf
                print(f"📅 {date.strftime('%Y-%m-%d')}: Switch to {best_etf} (Score: {best_etf_data['combined_score']:.3f}, Trend: {best_etf_data['monthly_trend']:.3f})")
            
            # Record portfolio value
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_etf': current_etf
            })
        
        # Calculate final performance
        final_value = portfolio_value
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate SPY benchmark
        spy_data = backtest_data[backtest_data['symbol'] == 'UPRO']  # Use UPRO as 3X SPY proxy
        if len(spy_data) > 0:
            spy_start = spy_data.iloc[0]['close']
            spy_end = spy_data.iloc[-1]['close']
            spy_return = (spy_end - spy_start) / spy_start * 100
        else:
            spy_return = 0
        
        # Store results
        self.results.update({
            'strategy_return': total_return,
            'spy_return': spy_return,
            'outperformance': total_return - spy_return,
            'final_portfolio_value': final_value,
            'initial_capital': initial_capital,
            'total_trades': len(trades),
            'trades': trades,
            'portfolio_history': portfolio_history,
            'best_model_combination': 'Ensemble (XGBoost + LightGBM + Random Forest)'
        })
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"💰 Strategy Return: {total_return:.2f}%")
        print(f"📈 UPRO (3X SPY) Return: {spy_return:.2f}%")
        print(f"🚀 Outperformance: {total_return - spy_return:.2f}%")
        print(f"💵 Final Portfolio Value: ${final_value:,.2f}")
        print(f"🔄 Total Trades: {len(trades)}")
        
        return self.results
    
    def save_results(self, filename='single_etf_switching_results.json'):
        """Save results to JSON file"""
        print(f"\n💾 Saving results to {filename}...")
        
        # Convert numpy types to native Python types for JSON serialization
        results_copy = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_copy[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_copy[key] = value.item()
            else:
                results_copy[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"✅ Results saved to {filename}")
    
    def run_complete_analysis(self):
        """Run the complete single ETF switching analysis"""
        print("🚀 Starting Single ETF Switching Strategy Analysis...")
        print("=" * 70)
        
        # Load and prepare data
        self.load_data()
        self.create_features()
        
        # Train ML models
        self.train_models()
        
        # Run backtest
        self.run_backtest()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 70)
        print("✅ Single ETF Switching Strategy Analysis Complete!")
        
        return self.results

def main():
    """Main execution function"""
    strategy = SingleETFSwitchingStrategy()
    results = strategy.run_complete_analysis()
    
    print("\n🎯 FINAL SUMMARY:")
    print(f"Strategy Return: {results['strategy_return']:.2f}%")
    print(f"Benchmark Return: {results['spy_return']:.2f}%")
    print(f"Outperformance: {results['outperformance']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Best Model: {results['best_model_combination']}")

if __name__ == "__main__":
    main()