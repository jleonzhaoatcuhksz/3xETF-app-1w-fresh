#!/usr/bin/env python3
"""
Fixed Benchmark Conservative Single ETF Switching Strategy
Starts with UPRO as benchmark, only switches when there's compelling evidence
Monthly rebalancing with conservative criteria
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

class FixedBenchmarkConservativeETFStrategy:
    """
    Conservative single ETF switching strategy that starts with UPRO benchmark
    - Always starts with UPRO position (benchmark)
    - Only switches monthly on first trading day
    - Requires high confidence and significant improvement over current position
    - Falls back to UPRO when no clear opportunities exist
    - Includes transaction costs
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
        self.benchmark_etf = 'UPRO'  # Default benchmark
        
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
    
    def create_conservative_features(self):
        """Create conservative features with medium-term time horizons"""
        print("🔧 Creating conservative features...")
        
        # Fill missing values
        self.data['close'] = self.data['close'].ffill()
        self.data['monthly_trend'] = self.data['monthly_trend'].fillna(0)
        self.data['sma_5d'] = self.data['sma_5d'].ffill()
        
        # Sort data by symbol and date
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate conservative technical indicators
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # Medium-term trend indicators
            symbol_data['returns_30d'] = symbol_data['close'].pct_change(periods=30).fillna(0)  # 1 month
            symbol_data['returns_60d'] = symbol_data['close'].pct_change(periods=60).fillna(0)  # 2 months
            symbol_data['returns_90d'] = symbol_data['close'].pct_change(periods=90).fillna(0)  # 3 months
            
            # Volatility indicators
            symbol_data['volatility_30d'] = symbol_data['close'].pct_change().rolling(30).std().fillna(0)
            symbol_data['volatility_60d'] = symbol_data['close'].pct_change().rolling(60).std().fillna(0)
            
            # Moving averages
            symbol_data['sma_20d'] = symbol_data['close'].rolling(20).mean().ffill()
            symbol_data['sma_50d'] = symbol_data['close'].rolling(50).mean().ffill()
            symbol_data['sma_100d'] = symbol_data['close'].rolling(100).mean().ffill()
            
            # Trend strength indicators
            symbol_data['trend_20_50'] = (symbol_data['sma_20d'] / symbol_data['sma_50d'] - 1).fillna(0)
            symbol_data['trend_50_100'] = (symbol_data['sma_50d'] / symbol_data['sma_100d'] - 1).fillna(0)
            
            # Price position relative to moving averages
            symbol_data['price_vs_sma20'] = (symbol_data['close'] / symbol_data['sma_20d'] - 1).fillna(0)
            symbol_data['price_vs_sma50'] = (symbol_data['close'] / symbol_data['sma_50d'] - 1).fillna(0)
            
            # Trend persistence (consistency of monthly trend)
            symbol_data['trend_persistence'] = symbol_data['monthly_trend'].rolling(6).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
            ).fillna(0)
            
            # Risk-adjusted returns (Sharpe-like ratio)
            symbol_data['risk_adj_return_30d'] = np.where(
                symbol_data['volatility_30d'] > 0,
                symbol_data['returns_30d'] / symbol_data['volatility_30d'],
                0
            )
            symbol_data['risk_adj_return_60d'] = np.where(
                symbol_data['volatility_60d'] > 0,
                symbol_data['returns_60d'] / symbol_data['volatility_60d'],
                0
            )
            
            # Update the main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        # Define feature columns for ML models
        self.feature_columns = [
            'monthly_trend', 'returns_30d', 'returns_60d', 'returns_90d',
            'volatility_30d', 'volatility_60d', 'trend_20_50', 'trend_50_100',
            'price_vs_sma20', 'price_vs_sma50', 'trend_persistence',
            'risk_adj_return_30d', 'risk_adj_return_60d'
        ]
        
        # Remove rows with missing features
        feature_data = self.data[self.feature_columns + ['symbol', 'date', 'close']]
        self.data = self.data[~feature_data[self.feature_columns].isna().any(axis=1)]
        
        print(f"✅ Created {len(self.feature_columns)} conservative features")
        print(f"📊 Data after feature creation: {len(self.data)} records")
        
    def prepare_ml_data(self):
        """Prepare data for machine learning models"""
        print("🤖 Preparing ML training data...")
        
        # Create target variable: identify top performers each month
        monthly_performance = []
        
        for date in self.data['date'].dt.to_period('M').unique():
            month_data = self.data[self.data['date'].dt.to_period('M') == date]
            
            if len(month_data) > 5:  # Need sufficient ETFs for comparison
                # Calculate forward returns for next month
                next_month_returns = []
                for symbol in month_data['symbol'].unique():
                    symbol_data = self.data[self.data['symbol'] == symbol].sort_values('date')
                    current_idx = symbol_data[symbol_data['date'].dt.to_period('M') == date].index
                    
                    if len(current_idx) > 0:
                        current_price = symbol_data.loc[current_idx[-1], 'close']
                        
                        # Find price 30 days later
                        future_data = symbol_data[symbol_data['date'] > symbol_data.loc[current_idx[-1], 'date']]
                        if len(future_data) >= 20:  # At least 20 days of data
                            future_price = future_data.iloc[19]['close']  # ~30 day return
                            forward_return = (future_price - current_price) / current_price
                            next_month_returns.append({
                                'symbol': symbol,
                                'forward_return': forward_return,
                                'date': date
                            })
                
                if len(next_month_returns) > 0:
                    # Identify top performer
                    returns_df = pd.DataFrame(next_month_returns)
                    top_performer = returns_df.loc[returns_df['forward_return'].idxmax(), 'symbol']
                    
                    # Mark top performer in the data
                    month_mask = (self.data['date'].dt.to_period('M') == date)
                    self.data.loc[month_mask, 'is_best_choice'] = 0
                    self.data.loc[month_mask & (self.data['symbol'] == top_performer), 'is_best_choice'] = 1
        
        # Prepare features and targets
        valid_data = self.data.dropna(subset=['is_best_choice'])
        
        X = valid_data[self.feature_columns].values
        y_best = valid_data['is_best_choice'].values
        
        # Split data
        X_train, X_test, y_best_train, y_best_test = train_test_split(
            X, y_best, test_size=0.3, random_state=42, stratify=y_best
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        print(f"✅ Training data prepared: {len(X_train)} training, {len(X_test)} testing samples")
        
        return X_train_scaled, X_test_scaled, y_best_train, y_best_test
    
    def train_conservative_models(self):
        """Train conservative ML models"""
        print("🎯 Training Conservative ML Models...")
        
        X_train_scaled, X_test_scaled, y_best_train, y_best_test = self.prepare_ml_data()
        
        # Train XGBoost model
        print("🎯 Training Conservative XGBoost Model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_best_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM model
        print("🎯 Training Conservative LightGBM Model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_best_train)
        self.models['lightgbm'] = lgb_model
        
        # Train Random Forest model
        print("🎯 Training Conservative Random Forest Model...")
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_best_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        self.evaluate_individual_models(X_test_scaled, y_best_test)
        
        print("✅ All conservative models trained successfully!")
    
    def evaluate_individual_models(self, X_test, y_test):
        """Evaluate individual model performance"""
        print("\n📈 Evaluating Model Performance...")
        
        model_performance = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            model_performance[f'{model_name}_accuracy'] = accuracy
            print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        self.results['model_performance'] = model_performance
    
    def run_benchmark_backtests(self, start_date='2020-01-01', initial_capital=10000):
        """Run backtests starting with UPRO benchmark for each ML algorithm"""
        print(f"\n🚀 Running BENCHMARK backtests from {start_date} with ${initial_capital:,} initial capital...")
        print(f"📊 Starting with {self.benchmark_etf} as benchmark position")
        
        # Filter data for backtest period
        backtest_data = self.data[self.data['date'] >= start_date].copy()
        backtest_data = backtest_data.sort_values('date')
        
        # Get first trading day of each month
        backtest_data['year_month'] = backtest_data['date'].dt.to_period('M')
        monthly_dates = backtest_data.groupby('year_month')['date'].min().values
        
        print(f"📅 Monthly rebalancing dates: {len(monthly_dates)} months")
        
        # Calculate UPRO benchmark performance
        upro_data = backtest_data[backtest_data['symbol'] == self.benchmark_etf].copy()
        if len(upro_data) > 0:
            upro_start_price = upro_data.iloc[0]['close']
            upro_end_price = upro_data.iloc[-1]['close']
            upro_total_return = (upro_end_price - upro_start_price) / upro_start_price
            print(f"📈 {self.benchmark_etf} benchmark total return: {upro_total_return:.2%}")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Running {model_name.upper()} Benchmark Strategy...")
            
            # Initialize portfolio with UPRO benchmark
            portfolio_value = initial_capital
            current_etf = self.benchmark_etf  # START WITH UPRO!
            trades = []
            portfolio_history = []
            
            for i, rebalance_date in enumerate(monthly_dates[6:]):  # Skip first 6 months for feature stability
                
                # Get data for current month
                current_month_data = backtest_data[backtest_data['date'] == rebalance_date]
                
                if len(current_month_data) == 0:
                    continue
                
                # Update portfolio value from previous position
                if i > 0:
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
                
                # Find the best ETF candidates
                best_scores = []
                for j, row in enumerate(valid_data):
                    if j < len(predictions):
                        confidence = predictions[j][1]  # Probability of being best choice
                        best_scores.append({
                            'symbol': row['symbol'],
                            'confidence': confidence,
                            'monthly_trend': row['monthly_trend'],
                            'returns_30d': row['returns_30d'],
                            'returns_60d': row['returns_60d'],
                            'trend_persistence': row['trend_persistence'],
                            'risk_adj_return_30d': row['risk_adj_return_30d'],
                            'volatility_30d': row['volatility_30d'],
                            'close': row['close']
                        })
                
                if len(best_scores) == 0:
                    continue
                
                # Sort by confidence and get the best candidate
                best_scores.sort(key=lambda x: x['confidence'], reverse=True)
                best_candidate = best_scores[0]
                
                # Get current ETF data for comparison
                current_etf_data = next((x for x in best_scores if x['symbol'] == current_etf), None)
                
                # Conservative switching criteria
                should_switch = False
                
                if current_etf_data and best_candidate['symbol'] != current_etf:
                    # Only switch if there's significant improvement
                    confidence_improvement = best_candidate['confidence'] - current_etf_data['confidence']
                    trend_improvement = best_candidate['monthly_trend'] - current_etf_data['monthly_trend']
                    performance_improvement = best_candidate['returns_60d'] - current_etf_data['returns_60d']
                    
                    # Conservative switching criteria
                    if (confidence_improvement > 0.15 and  # Meaningful confidence improvement
                        trend_improvement > 0.05 and     # Positive trend improvement
                        performance_improvement > 0.10 and  # Performance improvement
                        best_candidate['confidence'] > 0.7 and  # Reasonable absolute confidence
                        best_candidate['monthly_trend'] > 0.02 and  # Positive trend
                        best_candidate['trend_persistence'] > 0.5 and  # Some trend consistency
                        current_etf_data['monthly_trend'] < 0.02):  # Current ETF is weak
                        should_switch = True
                
                # Execute trade if switching
                if should_switch:
                    
                    # Apply transaction cost
                    portfolio_value *= (1 - self.transaction_cost)
                    
                    # Record trade
                    trades.append({
                        'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                        'from_etf': current_etf,
                        'to_etf': best_candidate['symbol'],
                        'confidence': float(best_candidate['confidence']),
                        'monthly_trend': float(best_candidate['monthly_trend']),
                        'returns_60d': float(best_candidate['returns_60d']),
                        'trend_persistence': float(best_candidate['trend_persistence']),
                        'portfolio_value': float(portfolio_value)
                    })
                    
                    # Update current ETF
                    current_etf = best_candidate['symbol']
                    
                    print(f"📅 {rebalance_date.strftime('%Y-%m-%d')}: Switch to {current_etf} "
                          f"(conf: {best_candidate['confidence']:.3f}, "
                          f"trend: {best_candidate['monthly_trend']:.3f})")
                
                # Record portfolio history
                portfolio_history.append({
                    'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                    'etf': current_etf,
                    'portfolio_value': float(portfolio_value)
                })
            
            # Calculate final performance
            total_return = (portfolio_value - initial_capital) / initial_capital
            num_trades = len(trades)
            transaction_costs = num_trades * self.transaction_cost * portfolio_value
            
            # Store results
            results[model_name] = {
                'total_return': total_return,
                'final_value': portfolio_value,
                'num_trades': num_trades,
                'transaction_costs': transaction_costs,
                'trades': trades,
                'portfolio_history': portfolio_history
            }
            
            print(f"✅ {model_name.upper()} Results:")
            print(f"   💰 Total Return: {total_return:.2%}")
            print(f"   💵 Final Value: ${portfolio_value:,.2f}")
            print(f"   🔄 Number of Trades: {num_trades}")
            print(f"   💸 Transaction Costs: ${transaction_costs:.2f}")
            
            if len(upro_data) > 0:
                excess_return = total_return - upro_total_return
                print(f"   📊 Excess vs {self.benchmark_etf}: {excess_return:.2%}")
        
        self.results['backtests'] = results
        return results
    
    def save_results(self, filename='benchmark_conservative_results.json'):
        """Save results to JSON file"""
        print(f"\n💾 Saving results to {filename}...")
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("✅ Results saved successfully!")

def main():
    """Main execution function"""
    print("🚀 Starting Fixed Benchmark Conservative ETF Switching Strategy...")
    
    # Initialize strategy
    strategy = FixedBenchmarkConservativeETFStrategy()
    
    # Load and prepare data
    strategy.load_data()
    strategy.create_conservative_features()
    
    # Train models
    strategy.train_conservative_models()
    
    # Run backtests starting with UPRO benchmark
    results = strategy.run_benchmark_backtests(start_date='2020-01-01')
    
    # Save results
    strategy.save_results()
    
    print("🎉 Fixed Benchmark Conservative Strategy completed successfully!")
    
    return strategy, results

if __name__ == "__main__":
    strategy, results = main()