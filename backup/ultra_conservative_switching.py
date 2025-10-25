#!/usr/bin/env python3
"""
Ultra Conservative Single ETF Switching Strategy
Minimal switching with very strict criteria to avoid over-trading
Monthly rebalancing only with high confidence requirements
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

class UltraConservativeETFStrategy:
    """
    Ultra conservative single ETF switching strategy
    - Only switches monthly on first trading day
    - Requires very high confidence (>0.95)
    - Requires persistent trends (3+ months)
    - Includes transaction costs
    - Focus on risk-adjusted returns
    """
    
    def __init__(self, db_path='etf_data.db'):
        self.db_path = db_path
        self.data = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.results = {}
        self.transaction_cost = 0.001  # 0.1% per trade
        
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
    
    def create_ultra_conservative_features(self):
        """Create ultra conservative features with longer time horizons"""
        print("🔧 Creating ultra conservative features...")
        
        # Fill missing values
        self.data['close'] = self.data['close'].ffill()
        self.data['monthly_trend'] = self.data['monthly_trend'].fillna(0)
        self.data['sma_5d'] = self.data['sma_5d'].ffill()
        
        # Sort data by symbol and date
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate ultra conservative technical indicators
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # Long-term trend indicators (much longer periods)
            symbol_data['returns_60d'] = symbol_data['close'].pct_change(periods=60).fillna(0)  # 3 months
            symbol_data['returns_120d'] = symbol_data['close'].pct_change(periods=120).fillna(0)  # 6 months
            
            # Long-term volatility (more stable)
            symbol_data['volatility_60d'] = symbol_data['close'].pct_change().rolling(60).std().fillna(0)
            
            # Long-term moving averages
            symbol_data['sma_50d'] = symbol_data['close'].rolling(50).mean().ffill()
            symbol_data['sma_100d'] = symbol_data['close'].rolling(100).mean().ffill()
            symbol_data['sma_200d'] = symbol_data['close'].rolling(200).mean().ffill()
            
            # Trend strength indicators
            symbol_data['trend_50_100'] = (symbol_data['sma_50d'] / symbol_data['sma_100d'] - 1).fillna(0)
            symbol_data['trend_100_200'] = (symbol_data['sma_100d'] / symbol_data['sma_200d'] - 1).fillna(0)
            
            # Price position relative to long-term averages
            symbol_data['price_vs_sma50'] = (symbol_data['close'] / symbol_data['sma_50d'] - 1).fillna(0)
            symbol_data['price_vs_sma200'] = (symbol_data['close'] / symbol_data['sma_200d'] - 1).fillna(0)
            
            # Trend persistence (how long has trend been consistent)
            symbol_data['trend_persistence'] = 0
            for i in range(60, len(symbol_data)):  # Start after 60 days
                recent_trends = symbol_data['monthly_trend'].iloc[i-60:i]
                if len(recent_trends) > 0:
                    positive_count = (recent_trends > 0).sum()
                    symbol_data.iloc[i, symbol_data.columns.get_loc('trend_persistence')] = positive_count / len(recent_trends)
            
            # Update main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        # Create cross-sectional features
        self.data = self.create_cross_sectional_features()
        
        # Create ultra conservative targets
        self.data = self.create_ultra_conservative_targets()
        
        # Select ultra conservative feature columns
        self.feature_columns = [
            'close', 'monthly_trend', 'returns_60d', 'returns_120d',
            'volatility_60d', 'sma_50d', 'sma_100d', 'sma_200d',
            'trend_50_100', 'trend_100_200', 'price_vs_sma50', 'price_vs_sma200',
            'trend_persistence', 'relative_strength_60d', 'rank_monthly_trend',
            'sector_encoded', 'category_encoded'
        ]
        
        print(f"✅ Created {len(self.feature_columns)} ultra conservative features")
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
            
            # Rank ETFs by monthly trend
            group['rank_monthly_trend'] = group['monthly_trend'].rank(ascending=False, method='min')
            
            # Long-term relative strength (60-day)
            median_60d = group['returns_60d'].median()
            group['relative_strength_60d'] = group['returns_60d'] - median_60d
            
            relative_features.append(group)
        
        result_df = pd.concat(relative_features).sort_values(['symbol', 'date'])
        return result_df
    
    def create_ultra_conservative_targets(self):
        """Create ultra conservative target variables"""
        print("🎯 Creating ultra conservative target variables...")
        
        # Sort by date
        self.data = self.data.sort_values('date')
        
        # For each date, identify ultra conservative switching targets
        date_groups = self.data.groupby('date')
        
        target_data = []
        for date, group in date_groups:
            group = group.copy()
            
            # Handle missing values
            valid_trends = group['monthly_trend'].dropna()
            valid_60d = group['returns_60d'].dropna()
            
            if len(valid_trends) == 0 or len(valid_60d) == 0:
                continue
            
            # Ultra conservative best ETF selection (top 5% only)
            top_threshold_monthly = valid_trends.quantile(0.95)  # Top 5%
            top_threshold_60d = valid_60d.quantile(0.95)  # Top 5%
            
            # Must be in top 5% for both monthly and 60-day returns
            top_etfs = group[
                (group['monthly_trend'] >= top_threshold_monthly) &
                (group['returns_60d'] >= top_threshold_60d) &
                (group['trend_persistence'] > 0.7) &  # 70% of recent trends positive
                (group['volatility_60d'] < group['volatility_60d'].quantile(0.5))  # Below median volatility
            ]
            
            if len(top_etfs) > 0:
                # Among ultra-qualified ETFs, choose the one with best risk-adjusted performance
                top_etfs = top_etfs.copy()
                top_etfs['risk_adjusted_return'] = top_etfs['returns_60d'] / (top_etfs['volatility_60d'] + 0.001)
                best_idx = top_etfs['risk_adjusted_return'].idxmax()
                best_etf = group.loc[best_idx, 'symbol']
                
                # Create binary target: 1 if this ETF is the ultra-conservative best choice
                group['is_ultra_best'] = (group['symbol'] == best_etf).astype(int)
                
                # More conservative switching signal (top 5% only with persistence)
                group['should_ultra_switch'] = (
                    (group['monthly_trend'] > top_threshold_monthly) &
                    (group['returns_60d'] > top_threshold_60d) &
                    (group['trend_persistence'] > 0.7)
                ).fillna(0).astype(int)
            else:
                # No clear ultra-conservative winner
                group['is_ultra_best'] = 0
                group['should_ultra_switch'] = 0
            
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
        ml_data = self.data[self.feature_columns + ['is_ultra_best', 'should_ultra_switch', 'target_etf_encoded']].dropna()
        
        X = ml_data[self.feature_columns]
        y_best = ml_data['is_ultra_best']
        y_switch = ml_data['should_ultra_switch']
        y_target = ml_data['target_etf_encoded']
        
        print(f"✅ Prepared {len(X)} samples with {len(self.feature_columns)} features")
        print(f"📊 Ultra best choice distribution: {y_best.value_counts().to_dict()}")
        print(f"📊 Ultra switch signal distribution: {y_switch.value_counts().to_dict()}")
        
        return X, y_best, y_switch, y_target
    
    def train_individual_models(self):
        """Train individual ML models with ultra conservative parameters"""
        print("🤖 Training ultra conservative ML models...")
        
        X, y_best, y_switch, y_target = self.prepare_ml_data()
        
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_best_train, y_best_test = y_best.iloc[:split_idx], y_best.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Train XGBoost model (ultra conservative parameters)
        print("\n🎯 Training Ultra Conservative XGBoost Model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=30,  # Even fewer trees
            max_depth=3,      # Very shallow trees
            learning_rate=0.01,  # Very slow learning
            subsample=0.7,    # More regularization
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_best_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM model (ultra conservative parameters)
        print("🎯 Training Ultra Conservative LightGBM Model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_best_train)
        self.models['lightgbm'] = lgb_model
        
        # Train Random Forest model (ultra conservative parameters)
        print("🎯 Training Ultra Conservative Random Forest Model...")
        rf_model = RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            min_samples_split=20,  # Very conservative splitting
            min_samples_leaf=10,   # Large leaf sizes
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_best_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        self.evaluate_individual_models(X_test_scaled, y_best_test)
        
        print("✅ All ultra conservative models trained successfully!")
    
    def evaluate_individual_models(self, X_test, y_test):
        """Evaluate individual model performance"""
        print("\n📈 Evaluating Ultra Conservative Model Performance...")
        
        model_performance = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        self.results['model_performance'] = model_performance
    
    def run_monthly_backtests(self, start_date='2020-01-01', initial_capital=10000):
        """Run monthly backtests for each ML algorithm"""
        print(f"\n🚀 Running MONTHLY backtests from {start_date} with ${initial_capital:,} initial capital...")
        
        # Filter data for backtest period
        backtest_data = self.data[self.data['date'] >= start_date].copy()
        backtest_data = backtest_data.sort_values('date')
        
        # Get first trading day of each month
        backtest_data['year_month'] = backtest_data['date'].dt.to_period('M')
        monthly_dates = backtest_data.groupby('year_month')['date'].min().values
        
        print(f"📅 Monthly rebalancing dates: {len(monthly_dates)} months")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Running {model_name.upper()} Monthly Strategy...")
            
            # Initialize portfolio for this model
            portfolio_value = initial_capital
            current_etf = None
            trades = []
            portfolio_history = []
            
            for i, rebalance_date in enumerate(monthly_dates[6:]):  # Skip first 6 months for feature stability
                
                # Get data for current month
                current_month_data = backtest_data[backtest_data['date'] == rebalance_date]
                
                if len(current_month_data) == 0:
                    continue
                
                # Prepare features for prediction
                valid_data = []
                for _, row in current_month_data.iterrows():
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
                        confidence = predictions[j][1]  # Probability of being ultra best choice
                        best_scores.append({
                            'symbol': row['symbol'],
                            'confidence': confidence,
                            'monthly_trend': row['monthly_trend'],
                            'returns_60d': row['returns_60d'],
                            'trend_persistence': row['trend_persistence'],
                            'volatility_60d': row['volatility_60d'],
                            'close': row['close']
                        })
                
                if len(best_scores) == 0:
                    continue
                
                # Sort by confidence and get the best
                best_scores.sort(key=lambda x: x['confidence'], reverse=True)
                best_candidate = best_scores[0]
                
                # Ultra conservative switching criteria
                should_switch = False
                
                if current_etf is None:
                    # Initial investment - ultra strict criteria
                    if (best_candidate['confidence'] > 0.95 and  # Ultra high confidence
                        best_candidate['monthly_trend'] > 0.10 and  # Very strong positive trend
                        best_candidate['returns_60d'] > 0.15 and  # Strong 60-day performance
                        best_candidate['trend_persistence'] > 0.8):  # Very consistent trend
                        should_switch = True
                else:
                    # Check current ETF performance first
                    current_etf_data = next((x for x in best_scores if x['symbol'] == current_etf), None)
                    
                    if current_etf_data and best_candidate:
                        # Only switch if there's overwhelming evidence
                        confidence_improvement = best_candidate['confidence'] - current_etf_data['confidence']
                        trend_improvement = best_candidate['monthly_trend'] - current_etf_data['monthly_trend']
                        performance_improvement = best_candidate['returns_60d'] - current_etf_data['returns_60d']
                        
                        # Ultra strict switching criteria
                        if (confidence_improvement > 0.3 and  # Massive confidence improvement
                            trend_improvement > 0.10 and     # Significant trend improvement
                            performance_improvement > 0.20 and  # Significant performance improvement
                            best_candidate['confidence'] > 0.95 and  # Ultra high absolute confidence
                            best_candidate['monthly_trend'] > 0.15 and  # Very strong positive trend
                            best_candidate['trend_persistence'] > 0.8 and  # Very consistent trend
                            current_etf_data['monthly_trend'] < 0.05):  # Current ETF is weak
                            should_switch = True
                
                # Update portfolio value from previous position
                if current_etf is not None and i > 0:
                    prev_month = monthly_dates[i + 6 - 1]
                    prev_data = backtest_data[
                        (backtest_data['date'] == prev_month) & 
                        (backtest_data['symbol'] == current_etf)
                    ]
                    current_data = backtest_data[
                        (backtest_data['date'] == rebalance_date) & 
                        (backtest_data['symbol'] == current_etf)
                    ]
                    
                    if len(prev_data) > 0 and len(current_data) > 0:
                        prev_price = prev_data.iloc[0]['close']
                        current_price = current_data.iloc[0]['close']
                        
                        # Update portfolio value
                        price_change = (current_price - prev_price) / prev_price
                        portfolio_value *= (1 + price_change)
                
                # Execute trade if switching
                if should_switch and best_candidate['symbol'] != current_etf:
                    
                    # Apply transaction cost
                    portfolio_value *= (1 - self.transaction_cost)
                    
                    # Record trade
                    trades.append({
                        'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                        'from_etf': current_etf,
                        'to_etf': best_candidate['symbol'],
                        'confidence': best_candidate['confidence'],
                        'monthly_trend': best_candidate['monthly_trend'],
                        'returns_60d': best_candidate['returns_60d'],
                        'trend_persistence': best_candidate['trend_persistence'],
                        'portfolio_value': portfolio_value
                    })
                    
                    current_etf = best_candidate['symbol']
                    print(f"📅 {model_name} - {pd.to_datetime(rebalance_date).strftime('%Y-%m-%d')}: Switch to {best_candidate['symbol']} (Conf: {best_candidate['confidence']:.3f}, Trend: {best_candidate['monthly_trend']:.3f}, 60d: {best_candidate['returns_60d']:.3f})")
                
                # Record portfolio value
                portfolio_history.append({
                    'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
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
                'portfolio_history': portfolio_history,
                'annualized_return': total_return / 5,  # Approximate annualized return
                'trades_per_year': len(trades) / 5
            }
            
            print(f"\n📊 {model_name.upper()} MONTHLY RESULTS:")
            print(f"💰 Strategy Return: {total_return:.2f}%")
            print(f"📈 UPRO Return: {upro_return:.2f}%")
            print(f"🚀 Outperformance: {total_return - upro_return:.2f}%")
            print(f"💵 Final Value: ${final_value:,.2f}")
            print(f"🔄 Total Trades: {len(trades)}")
            print(f"📊 Trades per Year: {len(trades) / 5:.1f}")
        
        self.results['monthly_strategies'] = results
        return results
    
    def save_results(self, filename='ultra_conservative_results.json'):
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
        """Run the complete ultra conservative switching analysis"""
        print("🚀 Starting Ultra Conservative Single ETF Switching Strategy Analysis...")
        print("=" * 80)
        
        # Load and prepare data
        self.load_data()
        self.create_ultra_conservative_features()
        
        # Train individual models
        self.train_individual_models()
        
        # Run monthly backtests
        self.run_monthly_backtests()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("✅ Ultra Conservative Single ETF Switching Strategy Analysis Complete!")
        
        return self.results

def main():
    """Main execution function"""
    strategy = UltraConservativeETFStrategy()
    results = strategy.run_complete_analysis()
    
    print("\n🎯 FINAL ULTRA CONSERVATIVE COMPARISON:")
    strategies = results['monthly_strategies']
    
    for model_name, model_results in strategies.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Strategy Return: {model_results['strategy_return']:.2f}%")
        print(f"  UPRO Return: {model_results['upro_return']:.2f}%")
        print(f"  Outperformance: {model_results['outperformance']:.2f}%")
        print(f"  Total Trades: {model_results['total_trades']}")
        print(f"  Trades per Year: {model_results['trades_per_year']:.1f}")
        print(f"  Annualized Return: {model_results['annualized_return']:.2f}%")
    
    # Find best performing strategy
    if len(strategies) > 0:
        best_strategy = max(strategies.items(), key=lambda x: x[1]['outperformance'])
        print(f"\n🏆 BEST ULTRA CONSERVATIVE STRATEGY: {best_strategy[0].upper()}")
        print(f"Outperformance: {best_strategy[1]['outperformance']:.2f}%")

if __name__ == "__main__":
    main()