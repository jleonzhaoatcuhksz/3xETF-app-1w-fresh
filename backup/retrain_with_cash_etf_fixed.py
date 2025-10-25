#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedETFSwitchingStrategy:
    def __init__(self, db_path='etf_data.db'):
        self.db_path = db_path
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def load_data(self):
        """Load all ETF data including the new VMOT cash ETF"""
        conn = sqlite3.connect(self.db_path)
        
        # Load all price data
        query = """
        SELECT date, symbol, close, volume
        FROM prices 
        ORDER BY date, symbol
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 Loaded {len(df):,} records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🎯 Symbols: {sorted(df['symbol'].unique())}")
        print(f"💰 Now includes VMOT cash ETF for risk management!")
        
        return df
    
    def create_features(self, df):
        """Create enhanced features with VMOT cash option"""
        print("\n🔧 Creating enhanced features with cash option...")
        
        # Pivot data to have symbols as columns
        price_df = df.pivot(index='date', columns='symbol', values='close')
        volume_df = df.pivot(index='date', columns='symbol', values='volume')
        
        # Forward fill missing values
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        volume_df = volume_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate returns for all ETFs
        returns_df = price_df.pct_change()
        
        # Create feature dataset
        features_list = []
        
        for i in range(21, len(price_df)):  # Need 21 days for features
            date = price_df.index[i]
            feature_row = {'date': date}
            
            # For each ETF, calculate technical indicators
            for symbol in price_df.columns:
                prices = price_df[symbol].iloc[i-20:i+1]  # 21 days
                volumes = volume_df[symbol].iloc[i-20:i+1]
                returns = returns_df[symbol].iloc[i-20:i+1]
                
                # Technical indicators
                feature_row[f'{symbol}_sma_5'] = prices.iloc[-5:].mean()
                feature_row[f'{symbol}_sma_10'] = prices.iloc[-10:].mean()
                feature_row[f'{symbol}_sma_20'] = prices.iloc[-20:].mean()
                
                # Momentum indicators
                feature_row[f'{symbol}_momentum_5'] = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else 0
                feature_row[f'{symbol}_momentum_10'] = (prices.iloc[-1] / prices.iloc[-11] - 1) if len(prices) >= 11 else 0
                feature_row[f'{symbol}_momentum_20'] = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
                
                # Volatility
                feature_row[f'{symbol}_volatility'] = returns.iloc[-10:].std() if len(returns) >= 10 else 0
                
                # Volume indicators
                feature_row[f'{symbol}_volume_ratio'] = volumes.iloc[-1] / volumes.iloc[-5:].mean() if volumes.iloc[-5:].mean() > 0 else 1
                
                # Price position
                feature_row[f'{symbol}_price_position'] = (prices.iloc[-1] - prices.iloc[-20:].min()) / (prices.iloc[-20:].max() - prices.iloc[-20:].min()) if (prices.iloc[-20:].max() - prices.iloc[-20:].min()) > 0 else 0.5
            
            # Market-wide features
            all_returns = []
            for symbol in price_df.columns:
                if symbol != 'VMOT':  # Exclude cash from market indicators
                    ret = returns_df[symbol].iloc[i]
                    if not pd.isna(ret):
                        all_returns.append(ret)
            
            if all_returns:
                feature_row['market_avg_return'] = np.mean(all_returns)
                feature_row['market_volatility'] = np.std(all_returns)
                feature_row['market_positive_ratio'] = sum(1 for r in all_returns if r > 0) / len(all_returns)
            else:
                feature_row['market_avg_return'] = 0
                feature_row['market_volatility'] = 0
                feature_row['market_positive_ratio'] = 0.5
            
            features_list.append(feature_row)
        
        features_df = pd.DataFrame(features_list)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        print(f"✅ Created {len(features_df)} feature rows with {len(features_df.columns)-1} features")
        return features_df, price_df
    
    def create_targets(self, features_df, price_df):
        """Create balanced target labels for best ETF to hold"""
        print("\n🎯 Creating balanced target labels...")
        
        targets = []
        symbols = [col for col in price_df.columns]
        
        for i in range(len(features_df)):
            date = features_df.iloc[i]['date']
            
            # Find the date index in price_df
            try:
                date_idx = price_df.index.get_loc(date)
            except KeyError:
                targets.append('UPRO')  # Default to UPRO if date not found
                continue
            
            # Look ahead 5 days for returns
            if date_idx + 5 < len(price_df):
                future_returns = {}
                
                for symbol in symbols:
                    current_price = price_df[symbol].iloc[date_idx]
                    future_price = price_df[symbol].iloc[date_idx + 5]
                    
                    if pd.notna(current_price) and pd.notna(future_price) and current_price > 0:
                        future_returns[symbol] = (future_price / current_price) - 1
                    else:
                        future_returns[symbol] = -0.1  # Penalty for missing data
                
                # Choose the ETF with highest future return
                if future_returns:
                    best_etf = max(future_returns.keys(), key=lambda x: future_returns[x])
                    best_return = future_returns[best_etf]
                    
                    # More balanced risk management logic:
                    # 1. Choose VMOT only in extreme negative conditions
                    # 2. Allow some risk-taking for better learning
                    if best_return < -0.05:  # Only if best option loses more than 5%
                        # Check if market stress conditions exist
                        market_volatility = features_df.iloc[i].get('market_volatility', 0)
                        market_positive_ratio = features_df.iloc[i].get('market_positive_ratio', 0.5)
                        
                        # Choose cash only in high stress + negative outlook
                        if market_volatility > 0.03 and market_positive_ratio < 0.3:
                            best_etf = 'VMOT'
                    
                    targets.append(best_etf)
                else:
                    targets.append('UPRO')  # Default to UPRO
            else:
                targets.append('UPRO')  # Default to UPRO for end of data
        
        features_df['target'] = targets
        
        # Show target distribution
        target_counts = pd.Series(targets).value_counts()
        print(f"📊 Target distribution:")
        for symbol, count in target_counts.items():
            percentage = (count / len(targets)) * 100
            print(f"   {symbol}: {count:,} ({percentage:.1f}%)")
        
        # Ensure we have at least 2 classes for ML
        unique_classes = len(target_counts)
        if unique_classes < 2:
            print("⚠️  Warning: Only one target class found. Adding some diversity...")
            # Force some VMOT selections for balance
            n_vmot = max(50, len(targets) // 20)  # At least 5% VMOT
            vmot_indices = np.random.choice(len(targets), size=n_vmot, replace=False)
            for idx in vmot_indices:
                targets[idx] = 'VMOT'
            features_df['target'] = targets
            
            # Show updated distribution
            target_counts = pd.Series(targets).value_counts()
            print(f"📊 Updated target distribution:")
            for symbol, count in target_counts.items():
                percentage = (count / len(targets)) * 100
                print(f"   {symbol}: {count:,} ({percentage:.1f}%)")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Prepare data for ML training"""
        print("\n🔧 Preparing training data...")
        
        # Remove date column for training
        feature_cols = [col for col in features_df.columns if col not in ['date', 'target']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Training data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"🎯 Target classes: {sorted(y.unique())}")
        return X_scaled, y, X
    
    def train_models(self, X_scaled, y, X_original):
        """Train all ML models"""
        print("\n🤖 Training ML models with cash option...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_orig, X_test_orig, _, _ = train_test_split(
            X_original, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. XGBoost
        print("🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        
        # 2. LightGBM
        print("⚡ Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        
        # 3. Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Store models
        self.models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'random_forest': rf_model
        }
        
        # Print results
        print(f"\n📊 Model Training Results:")
        print(f"🚀 XGBoost Accuracy:     {xgb_accuracy:.3f} ({xgb_accuracy*100:.1f}%)")
        print(f"⚡ LightGBM Accuracy:    {lgb_accuracy:.3f} ({lgb_accuracy*100:.1f}%)")
        print(f"🌳 Random Forest Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        
        return {
            'xgboost_accuracy': xgb_accuracy,
            'lightgbm_accuracy': lgb_accuracy,
            'random_forest_accuracy': rf_accuracy
        }
    
    def backtest_strategy(self, features_df, price_df, model_name):
        """Backtest a specific model strategy"""
        print(f"\n📈 Backtesting {model_name.upper()} strategy...")
        
        model = self.models[model_name]
        initial_capital = 10000
        portfolio_value = initial_capital
        current_etf = 'UPRO'  # Start with UPRO
        shares_owned = 0
        cash = initial_capital
        trades = []
        portfolio_history = []
        transaction_cost = 10  # $10 per trade
        
        # Get features for prediction
        feature_cols = self.feature_columns
        X_features = features_df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)
        
        for i in range(len(features_df)):
            date = features_df.iloc[i]['date']
            
            # Get current price
            try:
                if current_etf in price_df.columns:
                    current_price = price_df.loc[date, current_etf]
                else:
                    current_price = price_df.loc[date, 'UPRO']  # Fallback
            except (KeyError, IndexError):
                continue
            
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Calculate current portfolio value
            if shares_owned > 0:
                portfolio_value = shares_owned * current_price + cash
            else:
                portfolio_value = cash
            
            # Make prediction
            if i < len(X_scaled):
                prediction = model.predict(X_scaled[i:i+1])[0]
                
                # Execute trade if prediction differs from current holding
                if prediction != current_etf and i > 20:  # Allow warm-up period
                    # Sell current position
                    if shares_owned > 0:
                        cash = shares_owned * current_price - transaction_cost
                        shares_owned = 0
                    
                    # Buy new position
                    if prediction in price_df.columns:
                        try:
                            new_price = price_df.loc[date, prediction]
                            if pd.notna(new_price) and new_price > 0:
                                shares_owned = (cash - transaction_cost) / new_price
                                cash = 0
                                
                                trades.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'action': f'Switch from {current_etf} to {prediction}',
                                    'from_etf': current_etf,
                                    'to_etf': prediction,
                                    'price': new_price,
                                    'shares': shares_owned,
                                    'portfolio_value': shares_owned * new_price,
                                    'transaction_cost': transaction_cost
                                })
                                
                                current_etf = prediction
                        except (KeyError, IndexError):
                            pass  # Skip if price not available
            
            # Record portfolio history
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_etf': current_etf,
                'shares_owned': shares_owned
            })
        
        # Final portfolio value
        if shares_owned > 0 and current_etf in price_df.columns:
            final_price = price_df[current_etf].iloc[-1]
            if pd.notna(final_price):
                final_portfolio_value = shares_owned * final_price + cash
            else:
                final_portfolio_value = portfolio_value
        else:
            final_portfolio_value = cash
        
        total_return = (final_portfolio_value / initial_capital) - 1
        num_trades = len(trades)
        total_transaction_costs = num_trades * transaction_cost
        
        print(f"✅ {model_name.upper()} Results:")
        print(f"   💰 Final Value: ${final_portfolio_value:,.2f}")
        print(f"   📈 Total Return: {total_return*100:+.1f}%")
        print(f"   🔄 Number of Trades: {num_trades}")
        print(f"   💸 Transaction Costs: ${total_transaction_costs}")
        
        return {
            'total_return': total_return,
            'final_value': final_portfolio_value,
            'num_trades': num_trades,
            'total_transaction_costs': total_transaction_costs,
            'trades': trades,
            'portfolio_history': portfolio_history
        }
    
    def run_complete_analysis(self):
        """Run complete ML analysis with cash ETF"""
        print("🚀 ENHANCED ML ETF SWITCHING STRATEGY WITH CASH OPTION")
        print("=" * 70)
        
        # Load and prepare data
        df = self.load_data()
        features_df, price_df = self.create_features(df)
        features_df = self.create_targets(features_df, price_df)
        
        # Prepare training data
        X_scaled, y, X_original = self.prepare_training_data(features_df)
        
        # Train models
        model_performance = self.train_models(X_scaled, y, X_original)
        
        # Backtest all strategies
        backtest_results = {}
        for model_name in ['xgboost', 'lightgbm', 'random_forest']:
            backtest_results[model_name] = self.backtest_strategy(features_df, price_df, model_name)
        
        # Save results
        results = {
            'model_performance': model_performance,
            'backtests': backtest_results,
            'training_info': {
                'total_samples': len(features_df),
                'features_count': len(self.feature_columns),
                'symbols_included': list(price_df.columns),
                'date_range': f"{features_df['date'].min()} to {features_df['date'].max()}",
                'cash_etf_included': 'VMOT' in price_df.columns
            }
        }
        
        with open('enhanced_switching_results_with_cash.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to 'enhanced_switching_results_with_cash.json'")
        
        # Print summary
        print(f"\n🎯 ENHANCED STRATEGY SUMMARY:")
        print("=" * 50)
        for model_name, result in backtest_results.items():
            return_pct = result['total_return'] * 100
            print(f"{model_name.upper():<15}: {return_pct:+7.1f}% (${result['final_value']:,.0f}, {result['num_trades']} trades)")
        
        return results

if __name__ == "__main__":
    strategy = EnhancedETFSwitchingStrategy()
    results = strategy.run_complete_analysis()