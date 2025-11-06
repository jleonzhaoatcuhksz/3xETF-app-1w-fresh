#!/usr/bin/env python3
"""
Prudent Single ETF Switching Strategy with Individual ML Algorithms
Uses XGBoost, LightGBM, and Random Forest separately with conservative switching
Focuses on reducing over-switching and improving risk-adjusted returns
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

class PrudentETFSwitchingStrategy:
    """
    Conservative single ETF switching strategy with individual ML algorithms
    Focuses on high-confidence switches with strict criteria to avoid over-trading
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
        """Create conservative feature set for ML models"""
        print("🔧 Creating conservative features for ML models...")
        
        # Fill missing values conservatively
        self.data['close'] = self.data['close'].ffill()
        self.data['monthly_trend'] = self.data['monthly_trend'].fillna(0)
        self.data['sma_5d'] = self.data['sma_5d'].ffill()
        
        # Sort data by symbol and date
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate conservative technical indicators
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # Price-based features (more stable)
            symbol_data['returns_5d'] = symbol_data['close'].pct_change(periods=5).fillna(0)
            symbol_data['returns_20d'] = symbol_data['close'].pct_change(periods=20).fillna(0)
            
            # Volatility features (longer periods for stability)
            symbol_data['volatility_20d'] = symbol_data['close'].pct_change().rolling(20).std().fillna(0)
            
            # Moving averages (longer periods)
            symbol_data['sma_20d'] = symbol_data['close'].rolling(20).mean().ffill()
            symbol_data['sma_50d'] = symbol_data['close'].rolling(50).mean().ffill()
            
            # Price position relative to moving averages
            symbol_data['price_vs_sma20'] = (symbol_data['close'] / symbol_data['sma_20d'] - 1).fillna(0)
            symbol_data['price_vs_sma50'] = (symbol_data['close'] / symbol_data['sma_50d'] - 1).fillna(0)
            
            # Trend strength (smoother)
            symbol_data['trend_strength'] = (symbol_data['sma_20d'] / symbol_data['sma_50d'] - 1).fillna(0)
            
            # Update main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        # Create cross-sectional features
        self.data = self.create_cross_sectional_features()
        
        # Create conservative targets
        self.data = self.create_conservative_targets()
        
        # Select conservative feature columns
        self.feature_columns = [
            'close', 'sma_5d', 'monthly_trend', 'returns_5d', 'returns_20d',
            'volatility_20d', 'sma_20d', 'sma_50d', 'price_vs_sma20', 'price_vs_sma50',
            'trend_strength', 'relative_strength', 'rank_monthly_trend',
            'sector_encoded', 'category_encoded'
        ]
        
        print(f"✅ Created {len(self.feature_columns)} conservative features")
        return self.data
    
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
            
            # Rank ETFs by monthly trend (more conservative ranking)
            group['rank_monthly_trend'] = group['monthly_trend'].rank(ascending=False, method='min')
            
            # Relative strength vs median (more robust than mean)
            median_monthly_trend = group['monthly_trend'].median()
            group['relative_strength'] = group['monthly_trend'] - median_monthly_trend
            
            relative_features.append(group)
        
        result_df = pd.concat(relative_features).sort_values(['symbol', 'date'])
        return result_df
    
    def create_conservative_targets(self):
        """Create conservative target variables for ML models"""
        print("🎯 Creating conservative target variables...")
        
        # Sort by date
        self.data = self.data.sort_values('date')
        
        # For each date, identify conservative switching targets
        date_groups = self.data.groupby('date')
        
        target_data = []
        for date, group in date_groups:
            group = group.copy()
            
            # Handle missing values
            valid_trends = group['monthly_trend'].dropna()
            if len(valid_trends) == 0:
                continue
            
            # More conservative best ETF selection (top 10% only)
            top_threshold = valid_trends.quantile(0.90)  # Top 10%
            top_etfs = group[group['monthly_trend'] >= top_threshold]
            
            if len(top_etfs) > 0:
                # Among top ETFs, choose the one with best risk-adjusted performance
                best_idx = top_etfs['monthly_trend'].idxmax()
                best_etf = group.loc[best_idx, 'symbol']
            else:
                # If no clear winner, skip this date
                continue
            
            # Create binary target: 1 if this ETF is the clear best choice, 0 otherwise
            group['is_best_choice'] = (group['symbol'] == best_etf).astype(int)
            
            # More conservative switching signal (top 20% only)
            switch_threshold = valid_trends.quantile(0.80)  # Top 20%
            group['should_switch'] = (group['monthly_trend'] > switch_threshold).fillna(0).astype(int)
            
            # Target ETF
            group['target_etf'] = group['symbol']
            
            target_data.append(group)
        
        if len(target_data) == 0:
            raise ValueError("No valid target data created.")
        
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
        y_best = ml_data['is_best_choice']
        y_switch = ml_data['should_switch']
        y_target = ml_data['target_etf_encoded']
        
        print(f"✅ Prepared {len(X)} samples with {len(self.feature_columns)} features")
        print(f"📊 Best choice distribution: {y_best.value_counts().to_dict()}")
        print(f"📊 Switch signal distribution: {y_switch.value_counts().to_dict()}")
        
        return X, y_best, y_switch, y_target
    
    def train_individual_models(self):
        """Train individual ML models with conservative parameters"""
        print("🤖 Training individual ML models...")
        
        X, y_best, y_switch, y_target = self.prepare_ml_data()
        
        # Split data chronologically
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
        
        # Train XGBoost model (conservative parameters)
        print("\n🎯 Training XGBoost Model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,  # Fewer trees to reduce overfitting
            max_depth=4,      # Shallower trees
            learning_rate=0.05,  # Slower learning
            subsample=0.8,    # Regularization
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_best_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM model (conservative parameters)
        print("🎯 Training LightGBM Model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_best_train)
        self.models['lightgbm'] = lgb_model
        
        # Train Random Forest model (conservative parameters)
        print("🎯 Training Random Forest Model...")
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=10,  # More conservative splitting
            min_samples_leaf=5,    # Larger leaf sizes
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_best_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        self.evaluate_individual_models(X_test_scaled, y_best_test)
        
        print("✅ All individual models trained successfully!")
    
    def evaluate_individual_models(self, X_test, y_test):
        """Evaluate individual model performance"""
        print("\n📈 Evaluating Individual Model Performance...")
        
        model_performance = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        self.results['model_performance'] = model_performance
    
    def run_individual_backtests(self, start_date='2020-01-01', initial_capital=10000):
        """Run individual backtests for each ML algorithm"""
        print(f"\n🚀 Running individual backtests from {start_date} with ${initial_capital:,} initial capital...")
        
        # Filter data for backtest period
        backtest_data = self.data[self.data['date'] >= start_date].copy()
        backtest_data = backtest_data.sort_values('date')
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Running {model_name.upper()} Strategy...")
            
            # Initialize portfolio for this model
            portfolio_value = initial_capital
            current_etf = None
            trades = []
            portfolio_history = []
            
            # Get unique dates
            dates = sorted(backtest_data['date'].unique())
            
            for i, date in enumerate(dates[60:]):  # Skip first 60 days for feature stability
                
                # Get data for current date
                current_date_data = backtest_data[backtest_data['date'] == date]
                
                if len(current_date_data) == 0:
                    continue
                
                # Prepare features for prediction
                valid_data = []
                for _, row in current_date_data.iterrows():
                    if not pd.isna(row[self.feature_columns]).any():
                        valid_data.append(row)
                
                if len(valid_data) == 0:
                    continue
                
                # Convert to arrays for prediction
                X_current = np.array([row[self.feature_columns].values for row in valid_data])
                X_current_scaled = self.scalers['main'].transform(X_current)
                
                # Get predictions
                predictions = model.predict_proba(X_current_scaled)
                
                # Find the best ETF with highest confidence
                best_scores = []
                for j, row in enumerate(valid_data):
                    if j < len(predictions):
                        confidence = predictions[j][1]  # Probability of being best choice
                        best_scores.append({
                            'symbol': row['symbol'],
                            'confidence': confidence,
                            'monthly_trend': row['monthly_trend'],
                            'close': row['close']
                        })
                
                if len(best_scores) == 0:
                    continue
                
                # Sort by confidence and get the best
                best_scores.sort(key=lambda x: x['confidence'], reverse=True)
                best_candidate = best_scores[0]
                
                # Very conservative switching criteria
                should_switch = False
                
                if current_etf is None:
                    # Initial investment
                    if (best_candidate['confidence'] > 0.8 and  # Very high confidence
                        best_candidate['monthly_trend'] > 0.05):  # Strong positive trend
                        should_switch = True
                else:
                    # Check if switching is worthwhile
                    current_etf_data = next((x for x in best_scores if x['symbol'] == current_etf), None)
                    
                    if current_etf_data and best_candidate:
                        confidence_improvement = best_candidate['confidence'] - current_etf_data['confidence']
                        trend_improvement = best_candidate['monthly_trend'] - current_etf_data['monthly_trend']
                        
                        # Very strict switching criteria
                        if (confidence_improvement > 0.2 and  # Significant confidence improvement
                            trend_improvement > 0.03 and     # Significant trend improvement
                            best_candidate['confidence'] > 0.8 and  # High absolute confidence
                            best_candidate['monthly_trend'] > 0.05 and  # Strong positive trend
                            current_etf_data['monthly_trend'] < 0.02):  # Current ETF is weak
                            should_switch = True
                
                # Execute trade if switching
                if should_switch and best_candidate['symbol'] != current_etf:
                    
                    # Calculate portfolio value change from previous position
                    if current_etf is not None:
                        # Find previous price
                        prev_date_idx = max(0, i - 1)
                        prev_date = dates[prev_date_idx + 60]
                        prev_data = backtest_data[
                            (backtest_data['date'] == prev_date) & 
                            (backtest_data['symbol'] == current_etf)
                        ]
                        
                        if len(prev_data) > 0:
                            prev_price = prev_data.iloc[0]['close']
                            current_price = current_etf_data['close'] if current_etf_data else prev_price
                            
                            # Update portfolio value
                            price_change = (current_price - prev_price) / prev_price
                            portfolio_value *= (1 + price_change)
                    
                    # Record trade
                    trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'from_etf': current_etf,
                        'to_etf': best_candidate['symbol'],
                        'confidence': best_candidate['confidence'],
                        'monthly_trend': best_candidate['monthly_trend'],
                        'portfolio_value': portfolio_value
                    })
                    
                    current_etf = best_candidate['symbol']
                    print(f"📅 {model_name} - {date.strftime('%Y-%m-%d')}: Switch to {best_candidate['symbol']} (Conf: {best_candidate['confidence']:.3f}, Trend: {best_candidate['monthly_trend']:.3f})")
                
                # Record portfolio value
                portfolio_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'current_etf': current_etf
                })
            
            # Calculate performance
            final_value = portfolio_value
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # Calculate benchmark (UPRO)
            upro_data = backtest_data[backtest_data['symbol'] == 'UPRO']
            if len(upro_data) > 0:
                upro_start = upro_data.iloc[0]['close']
                upro_end = upro_data.iloc[-1]['close']
                upro_return = (upro_end - upro_start) / upro_start * 100
            else:
                upro_return = 0
            
            # Store results for this model
            results[model_name] = {
                'strategy_return': total_return,
                'upro_return': upro_return,
                'outperformance': total_return - upro_return,
                'final_portfolio_value': final_value,
                'initial_capital': initial_capital,
                'total_trades': len(trades),
                'trades': trades,
                'portfolio_history': portfolio_history
            }
            
            print(f"\n📊 {model_name.upper()} RESULTS:")
            print(f"💰 Strategy Return: {total_return:.2f}%")
            print(f"📈 UPRO Return: {upro_return:.2f}%")
            print(f"🚀 Outperformance: {total_return - upro_return:.2f}%")
            print(f"💵 Final Value: ${final_value:,.2f}")
            print(f"🔄 Total Trades: {len(trades)}")
        
        self.results['individual_strategies'] = results
        return results
    
    def save_results(self, filename='prudent_switching_results.json'):
        """Save results to JSON file"""
        print(f"\n💾 Saving results to {filename}...")
        
        # Convert numpy types for JSON serialization
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
        """Run the complete prudent switching analysis"""
        print("🚀 Starting Prudent Single ETF Switching Strategy Analysis...")
        print("=" * 80)
        
        # Load and prepare data
        self.load_data()
        self.create_features()
        
        # Train individual models
        self.train_individual_models()
        
        # Run individual backtests
        self.run_individual_backtests()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("✅ Prudent Single ETF Switching Strategy Analysis Complete!")
        
        return self.results

def main():
    """Main execution function"""
    strategy = PrudentETFSwitchingStrategy()
    results = strategy.run_complete_analysis()
    
    print("\n🎯 FINAL COMPARISON:")
    strategies = results['individual_strategies']
    
    for model_name, model_results in strategies.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Strategy Return: {model_results['strategy_return']:.2f}%")
        print(f"  UPRO Return: {model_results['upro_return']:.2f}%")
        print(f"  Outperformance: {model_results['outperformance']:.2f}%")
        print(f"  Total Trades: {model_results['total_trades']}")
    
    # Find best performing strategy
    best_strategy = max(strategies.items(), key=lambda x: x[1]['outperformance'])
    print(f"\n🏆 BEST STRATEGY: {best_strategy[0].upper()}")
    print(f"Outperformance: {best_strategy[1]['outperformance']:.2f}%")

if __name__ == "__main__":
    main()