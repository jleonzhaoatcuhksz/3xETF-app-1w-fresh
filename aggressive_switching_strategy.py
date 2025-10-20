#!/usr/bin/env python3
"""
Aggressive Single ETF Switching Strategy
More risk-taking with relaxed switching criteria
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

class AggressiveSwitchingETFStrategy:
    """
    Aggressive single ETF switching strategy with relaxed risk criteria
    - Starts with UPRO as benchmark
    - More frequent switching with lower confidence thresholds
    - Takes calculated risks for higher potential returns
    - Monthly rebalancing with aggressive criteria
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
    
    def create_aggressive_features(self):
        """Create aggressive features with shorter time horizons for faster signals"""
        print("🔧 Creating aggressive features...")
        
        # Fill missing values
        self.data['close'] = self.data['close'].ffill()
        self.data['monthly_trend'] = self.data['monthly_trend'].fillna(0)
        self.data['sma_5d'] = self.data['sma_5d'].ffill()
        
        # Sort data by symbol and date
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate aggressive technical indicators
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # Short-term trend indicators for faster signals
            symbol_data['returns_5d'] = symbol_data['close'].pct_change(periods=5).fillna(0)
            symbol_data['returns_10d'] = symbol_data['close'].pct_change(periods=10).fillna(0)
            symbol_data['returns_20d'] = symbol_data['close'].pct_change(periods=20).fillna(0)
            symbol_data['returns_30d'] = symbol_data['close'].pct_change(periods=30).fillna(0)
            symbol_data['returns_60d'] = symbol_data['close'].pct_change(periods=60).fillna(0)
            
            # Short-term volatility for risk assessment
            symbol_data['volatility_10d'] = symbol_data['close'].pct_change().rolling(10).std().fillna(0)
            symbol_data['volatility_20d'] = symbol_data['close'].pct_change().rolling(20).std().fillna(0)
            symbol_data['volatility_30d'] = symbol_data['close'].pct_change().rolling(30).std().fillna(0)
            
            # Fast moving averages for trend detection
            symbol_data['sma_10d'] = symbol_data['close'].rolling(10).mean().ffill()
            symbol_data['sma_20d'] = symbol_data['close'].rolling(20).mean().ffill()
            symbol_data['sma_50d'] = symbol_data['close'].rolling(50).mean().ffill()
            
            # Aggressive trend indicators
            symbol_data['trend_5_20'] = (symbol_data['sma_5d'] / symbol_data['sma_20d'] - 1).fillna(0)
            symbol_data['trend_10_50'] = (symbol_data['sma_10d'] / symbol_data['sma_50d'] - 1).fillna(0)
            symbol_data['trend_20_50'] = (symbol_data['sma_20d'] / symbol_data['sma_50d'] - 1).fillna(0)
            
            # Price momentum indicators
            symbol_data['price_vs_sma5'] = (symbol_data['close'] / symbol_data['sma_5d'] - 1).fillna(0)
            symbol_data['price_vs_sma10'] = (symbol_data['close'] / symbol_data['sma_10d'] - 1).fillna(0)
            symbol_data['price_vs_sma20'] = (symbol_data['close'] / symbol_data['sma_20d'] - 1).fillna(0)
            
            # Short-term trend persistence (more responsive)
            symbol_data['trend_persistence_3m'] = symbol_data['monthly_trend'].rolling(3).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
            ).fillna(0)
            
            # Momentum indicators
            symbol_data['momentum_5d'] = symbol_data['close'] / symbol_data['close'].shift(5) - 1
            symbol_data['momentum_10d'] = symbol_data['close'] / symbol_data['close'].shift(10) - 1
            
            # Risk-adjusted returns (shorter periods)
            symbol_data['risk_adj_return_10d'] = np.where(
                symbol_data['volatility_10d'] > 0,
                symbol_data['returns_10d'] / symbol_data['volatility_10d'],
                0
            )
            symbol_data['risk_adj_return_20d'] = np.where(
                symbol_data['volatility_20d'] > 0,
                symbol_data['returns_20d'] / symbol_data['volatility_20d'],
                0
            )
            
            # Relative strength (vs average)
            all_symbols_data = self.data[self.data['date'].isin(symbol_data['date'])]
            avg_returns = all_symbols_data.groupby('date')['monthly_trend'].mean()
            symbol_data['relative_strength'] = symbol_data['monthly_trend'] - symbol_data['date'].map(avg_returns)
            
            # Update the main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        # Define feature columns for ML models (more aggressive features)
        self.feature_columns = [
            'monthly_trend', 'returns_5d', 'returns_10d', 'returns_20d', 'returns_30d', 'returns_60d',
            'volatility_10d', 'volatility_20d', 'volatility_30d',
            'trend_5_20', 'trend_10_50', 'trend_20_50',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
            'trend_persistence_3m', 'momentum_5d', 'momentum_10d',
            'risk_adj_return_10d', 'risk_adj_return_20d', 'relative_strength'
        ]
        
        # Remove rows with missing features
        feature_data = self.data[self.feature_columns + ['symbol', 'date', 'close']]
        self.data = self.data[~feature_data[self.feature_columns].isna().any(axis=1)]
        
        print(f"✅ Created {len(self.feature_columns)} aggressive features")
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
                    # Identify top performers (top 30% instead of just #1)
                    returns_df = pd.DataFrame(next_month_returns)
                    top_30_pct = int(len(returns_df) * 0.3) or 1
                    top_performers = returns_df.nlargest(top_30_pct, 'forward_return')['symbol'].tolist()
                    
                    # Mark top performers in the data
                    month_mask = (self.data['date'].dt.to_period('M') == date)
                    self.data.loc[month_mask, 'is_best_choice'] = 0
                    self.data.loc[month_mask & (self.data['symbol'].isin(top_performers)), 'is_best_choice'] = 1
        
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
    
    def train_aggressive_models(self):
        """Train aggressive ML models"""
        print("🎯 Training Aggressive ML Models...")
        
        X_train_scaled, X_test_scaled, y_best_train, y_best_test = self.prepare_ml_data()
        
        # Train XGBoost model (more aggressive parameters)
        print("🎯 Training Aggressive XGBoost Model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # More trees
            max_depth=6,       # Deeper trees
            learning_rate=0.15, # Faster learning
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_best_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM model (more aggressive)
        print("🎯 Training Aggressive LightGBM Model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.15,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_best_train)
        self.models['lightgbm'] = lgb_model
        
        # Train Random Forest model (more aggressive)
        print("🎯 Training Aggressive Random Forest Model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,       # Deeper trees
            min_samples_split=5, # Less conservative splitting
            min_samples_leaf=2,  # Smaller leaf sizes
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_best_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        self.evaluate_individual_models(X_test_scaled, y_best_test)
        
        print("✅ All aggressive models trained successfully!")
    
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
    
    def run_aggressive_backtests(self, start_date='2020-01-01', initial_capital=10000):
        """Run aggressive backtests with relaxed switching criteria"""
        print(f"\n🚀 Running AGGRESSIVE backtests from {start_date} with ${initial_capital:,} initial capital...")
        print(f"📊 Starting with {self.benchmark_etf} as benchmark position")
        
        # Filter data for backtest period
        backtest_data = self.data[self.data['date'] >= start_date].copy()
        backtest_data = backtest_data.sort_values('date')
        
        # Get first trading day of each month
        backtest_data['year_month'] = backtest_data['date'].dt.to_period('M')
        monthly_dates = backtest_data.groupby('year_month')['date'].min().values
        
        print(f"📅 Monthly rebalancing dates: {len(monthly_dates)} months")
        
        # Calculate UPRO benchmark performance for comparison
        upro_data = backtest_data[backtest_data['symbol'] == self.benchmark_etf].copy()
        if len(upro_data) > 0:
            upro_start_price = upro_data.iloc[0]['close']
            upro_end_price = upro_data.iloc[-1]['close']
            upro_total_return = (upro_end_price - upro_start_price) / upro_start_price
            print(f"📈 {self.benchmark_etf} benchmark total return: {upro_total_return:.2%}")
        else:
            upro_total_return = 0
            print(f"⚠️  No {self.benchmark_etf} data found for benchmark calculation")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Running {model_name.upper()} Aggressive Strategy...")
            
            # Initialize portfolio with UPRO benchmark
            current_etf = self.benchmark_etf  # START WITH UPRO!
            shares_owned = 0  # Will be set on first month
            portfolio_value = initial_capital
            trades = []
            portfolio_history = []
            
            for i, rebalance_date in enumerate(monthly_dates[3:]):  # Skip only 3 months (less conservative)
                
                # Get data for current month
                current_month_data = backtest_data[backtest_data['date'] == rebalance_date]
                
                if len(current_month_data) == 0:
                    continue
                
                # Get current ETF price
                current_etf_price_data = current_month_data[current_month_data['symbol'] == current_etf]
                if len(current_etf_price_data) == 0:
                    continue
                
                current_price = current_etf_price_data.iloc[0]['close']
                
                # Initialize shares on first month (buy UPRO with initial capital)
                if shares_owned == 0:
                    shares_owned = initial_capital / current_price
                    print(f"📅 {pd.to_datetime(rebalance_date).strftime('%Y-%m-%d')}: Initial investment in {current_etf} "
                          f"at ${current_price:.2f} - {shares_owned:.2f} shares")
                
                # Calculate current portfolio value
                portfolio_value = shares_owned * current_price
                
                # Prepare features for prediction
                valid_data = []
                for _, row in current_month_data.iterrows():
                    if not pd.isna(row[self.feature_columns]).any():
                        valid_data.append(row)
                
                if len(valid_data) == 0:
                    # Record portfolio history even if no prediction possible
                    portfolio_history.append({
                        'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                        'etf': current_etf,
                        'shares': float(shares_owned),
                        'price': float(current_price),
                        'portfolio_value': float(portfolio_value)
                    })
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
                            'returns_10d': row['returns_10d'],
                            'returns_20d': row['returns_20d'],
                            'returns_30d': row['returns_30d'],
                            'momentum_10d': row['momentum_10d'],
                            'trend_persistence_3m': row['trend_persistence_3m'],
                            'relative_strength': row['relative_strength'],
                            'risk_adj_return_20d': row['risk_adj_return_20d'],
                            'volatility_20d': row['volatility_20d'],
                            'close': row['close']
                        })
                
                if len(best_scores) == 0:
                    # Record portfolio history even if no candidates
                    portfolio_history.append({
                        'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                        'etf': current_etf,
                        'shares': float(shares_owned),
                        'price': float(current_price),
                        'portfolio_value': float(portfolio_value)
                    })
                    continue
                
                # Sort by confidence and get the best candidate
                best_scores.sort(key=lambda x: x['confidence'], reverse=True)
                best_candidate = best_scores[0]
                
                # Get current ETF data for comparison
                current_etf_data = next((x for x in best_scores if x['symbol'] == current_etf), None)
                
                # AGGRESSIVE switching criteria (much more relaxed)
                should_switch = False
                
                if current_etf_data and best_candidate['symbol'] != current_etf:
                    # Relaxed switching criteria for aggressive approach
                    confidence_improvement = best_candidate['confidence'] - current_etf_data['confidence']
                    trend_improvement = best_candidate['monthly_trend'] - current_etf_data['monthly_trend']
                    momentum_improvement = best_candidate['momentum_10d'] - current_etf_data['momentum_10d']
                    relative_strength_improvement = best_candidate['relative_strength'] - current_etf_data['relative_strength']
                    
                    # AGGRESSIVE switching criteria (much lower thresholds)
                    if (confidence_improvement > 0.05 and  # Low confidence improvement threshold
                        trend_improvement > 0.01 and      # Small trend improvement
                        best_candidate['confidence'] > 0.4 and  # Low absolute confidence threshold
                        best_candidate['monthly_trend'] > -0.05 and  # Allow slightly negative trends
                        (momentum_improvement > 0.02 or    # Either momentum improvement
                         relative_strength_improvement > 0.01 or  # Or relative strength improvement
                         best_candidate['relative_strength'] > 0.02)):  # Or strong relative strength
                        should_switch = True
                    
                    # Additional aggressive switching conditions
                    elif (best_candidate['confidence'] > 0.6 and  # High confidence alone
                          best_candidate['monthly_trend'] > 0.03):  # With decent trend
                        should_switch = True
                    
                    # Momentum-based switching
                    elif (best_candidate['momentum_10d'] > 0.05 and  # Strong momentum
                          best_candidate['relative_strength'] > 0.03 and  # Strong relative performance
                          current_etf_data['momentum_10d'] < 0.02):  # Current ETF losing momentum
                        should_switch = True
                
                # Execute trade if switching
                if should_switch:
                    # Sell current position (apply transaction cost)
                    portfolio_value *= (1 - self.transaction_cost)
                    
                    # Buy new ETF
                    new_price = best_candidate['close']
                    shares_owned = portfolio_value / new_price
                    
                    # Record trade
                    trades.append({
                        'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                        'from_etf': current_etf,
                        'to_etf': best_candidate['symbol'],
                        'from_price': float(current_price),
                        'to_price': float(new_price),
                        'shares_sold': float(portfolio_value / current_price),
                        'shares_bought': float(shares_owned),
                        'confidence': float(best_candidate['confidence']),
                        'monthly_trend': float(best_candidate['monthly_trend']),
                        'momentum_10d': float(best_candidate['momentum_10d']),
                        'relative_strength': float(best_candidate['relative_strength']),
                        'portfolio_value': float(portfolio_value)
                    })
                    
                    # Update current ETF
                    current_etf = best_candidate['symbol']
                    
                    print(f"📅 {pd.to_datetime(rebalance_date).strftime('%Y-%m-%d')}: Switch to {current_etf} "
                          f"at ${new_price:.2f} (conf: {best_candidate['confidence']:.3f}, "
                          f"trend: {best_candidate['monthly_trend']:.3f}, "
                          f"momentum: {best_candidate['momentum_10d']:.3f})")
                
                # Record portfolio history
                portfolio_history.append({
                    'date': pd.to_datetime(rebalance_date).strftime('%Y-%m-%d'),
                    'etf': current_etf,
                    'shares': float(shares_owned),
                    'price': float(current_price if not should_switch else best_candidate['close']),
                    'portfolio_value': float(portfolio_value)
                })
            
            # Calculate final performance
            final_portfolio_value = portfolio_value
            total_return = (final_portfolio_value - initial_capital) / initial_capital
            num_trades = len(trades)
            total_transaction_costs = num_trades * self.transaction_cost * initial_capital
            
            # Store results
            results[model_name] = {
                'total_return': total_return,
                'final_value': final_portfolio_value,
                'num_trades': num_trades,
                'total_transaction_costs': total_transaction_costs,
                'trades': trades,
                'portfolio_history': portfolio_history
            }
            
            print(f"✅ {model_name.upper()} Results:")
            print(f"   💰 Total Return: {total_return:.2%}")
            print(f"   💵 Final Value: ${final_portfolio_value:,.2f}")
            print(f"   🔄 Number of Trades: {num_trades}")
            print(f"   💸 Transaction Costs: ${total_transaction_costs:.2f}")
            
            if upro_total_return != 0:
                excess_return = total_return - upro_total_return
                print(f"   📊 Excess vs {self.benchmark_etf}: {excess_return:.2%}")
        
        self.results['backtests'] = results
        return results
    
    def save_results(self, filename='aggressive_switching_results.json'):
        """Save results to JSON file"""
        print(f"\n💾 Saving results to {filename}...")
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("✅ Results saved successfully!")

def main():
    """Main execution function"""
    print("🚀 Starting Aggressive Single ETF Switching Strategy...")
    
    # Initialize strategy
    strategy = AggressiveSwitchingETFStrategy()
    
    # Load and prepare data
    strategy.load_data()
    strategy.create_aggressive_features()
    
    # Train models
    strategy.train_aggressive_models()
    
    # Run backtests with aggressive criteria
    results = strategy.run_aggressive_backtests(start_date='2020-01-01')
    
    # Save results
    strategy.save_results()
    
    print("🎉 Aggressive Single ETF Switching Strategy completed successfully!")
    
    return strategy, results

if __name__ == "__main__":
    strategy, results = main()