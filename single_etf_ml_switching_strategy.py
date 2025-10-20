#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

class SingleETFMLSwitchingStrategy:
    """
    Single ETF switching strategy using ML algorithms to determine both:
    1. Which ETF to switch to (target selection)
    2. When to make the switch (timing)
    
    Uses monthly_trend column and technical indicators as features.
    """
    
    def __init__(self, db_path='etf_data.db'):
        self.db_path = db_path
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def load_data(self):
        """Load ETF data with monthly_trend column"""
        conn = sqlite3.connect(self.db_path)
        
        # Load price data including monthly_trend
        query = """
        SELECT date, symbol, close, volume, monthly_trend, sma_5d,
               open, high, low, adj_close
        FROM prices 
        WHERE date >= '2020-01-01'  -- Use recent data for better ML performance
        ORDER BY date, symbol
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 Loaded {len(df):,} records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🎯 Symbols: {sorted(df['symbol'].unique())}")
        print(f"📈 Monthly trend data available: {df['monthly_trend'].notna().sum():,} records")
        
        return df
    
    def create_features(self, df):
        """Create comprehensive features for each ETF including monthly_trend"""
        print("\n🔧 Creating ML features...")
        
        # Pivot data to have symbols as columns
        price_df = df.pivot(index='date', columns='symbol', values='close')
        volume_df = df.pivot(index='date', columns='symbol', values='volume')
        trend_df = df.pivot(index='date', columns='symbol', values='monthly_trend')
        sma_df = df.pivot(index='date', columns='symbol', values='sma_5d')
        
        # Forward fill missing values
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        volume_df = volume_df.fillna(method='ffill').fillna(method='bfill')
        trend_df = trend_df.fillna(method='ffill').fillna(0)  # Fill missing trends with 0
        sma_df = sma_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate returns
        returns_df = price_df.pct_change()
        
        # Create feature dataset
        features_list = []
        
        for i in range(30, len(price_df)):  # Need 30 days for comprehensive features
            date = price_df.index[i]
            feature_row = {'date': date}
            
            # For each ETF, calculate comprehensive features
            for symbol in price_df.columns:
                if symbol not in price_df.columns:
                    continue
                    
                prices = price_df[symbol].iloc[i-29:i+1]  # 30 days
                volumes = volume_df[symbol].iloc[i-29:i+1]
                returns = returns_df[symbol].iloc[i-29:i+1]
                
                # Monthly trend (most important feature)
                current_trend = trend_df[symbol].iloc[i] if symbol in trend_df.columns else 0
                feature_row[f'{symbol}_monthly_trend'] = current_trend if pd.notna(current_trend) else 0
                
                # Price-based features
                current_price = prices.iloc[-1]
                feature_row[f'{symbol}_price'] = current_price
                
                # Moving averages
                feature_row[f'{symbol}_sma_5'] = prices.iloc[-5:].mean()
                feature_row[f'{symbol}_sma_10'] = prices.iloc[-10:].mean()
                feature_row[f'{symbol}_sma_20'] = prices.iloc[-20:].mean()
                feature_row[f'{symbol}_sma_30'] = prices.iloc[-30:].mean()
                
                # Price position relative to moving averages
                sma_5 = feature_row[f'{symbol}_sma_5']
                sma_20 = feature_row[f'{symbol}_sma_20']
                if sma_5 > 0 and sma_20 > 0:
                    feature_row[f'{symbol}_price_vs_sma5'] = (current_price / sma_5) - 1
                    feature_row[f'{symbol}_price_vs_sma20'] = (current_price / sma_20) - 1
                    feature_row[f'{symbol}_sma5_vs_sma20'] = (sma_5 / sma_20) - 1
                else:
                    feature_row[f'{symbol}_price_vs_sma5'] = 0
                    feature_row[f'{symbol}_price_vs_sma20'] = 0
                    feature_row[f'{symbol}_sma5_vs_sma20'] = 0
                
                # Momentum indicators
                if len(prices) >= 5:
                    feature_row[f'{symbol}_momentum_5d'] = (prices.iloc[-1] / prices.iloc[-5] - 1) if prices.iloc[-5] > 0 else 0
                if len(prices) >= 10:
                    feature_row[f'{symbol}_momentum_10d'] = (prices.iloc[-1] / prices.iloc[-10] - 1) if prices.iloc[-10] > 0 else 0
                if len(prices) >= 20:
                    feature_row[f'{symbol}_momentum_20d'] = (prices.iloc[-1] / prices.iloc[-20] - 1) if prices.iloc[-20] > 0 else 0
                
                # Volatility
                if len(returns) >= 10:
                    feature_row[f'{symbol}_volatility_10d'] = returns.iloc[-10:].std()
                if len(returns) >= 20:
                    feature_row[f'{symbol}_volatility_20d'] = returns.iloc[-20:].std()
                
                # Volume indicators
                avg_volume = volumes.iloc[-10:].mean()
                if avg_volume > 0:
                    feature_row[f'{symbol}_volume_ratio'] = volumes.iloc[-1] / avg_volume
                else:
                    feature_row[f'{symbol}_volume_ratio'] = 1
                
                # Price range features
                high_20d = prices.iloc[-20:].max()
                low_20d = prices.iloc[-20:].min()
                if high_20d > low_20d:
                    feature_row[f'{symbol}_price_position'] = (current_price - low_20d) / (high_20d - low_20d)
                else:
                    feature_row[f'{symbol}_price_position'] = 0.5
            
            # Market-wide features
            all_returns = []
            all_trends = []
            all_volatilities = []
            
            for symbol in price_df.columns:
                ret = returns_df[symbol].iloc[i]
                if pd.notna(ret):
                    all_returns.append(ret)
                    
                trend = trend_df[symbol].iloc[i] if symbol in trend_df.columns else 0
                if pd.notna(trend):
                    all_trends.append(trend)
                    
                vol = returns_df[symbol].iloc[i-9:i+1].std()
                if pd.notna(vol):
                    all_volatilities.append(vol)
            
            # Market indicators
            if all_returns:
                feature_row['market_avg_return'] = np.mean(all_returns)
                feature_row['market_positive_ratio'] = sum(1 for r in all_returns if r > 0) / len(all_returns)
            else:
                feature_row['market_avg_return'] = 0
                feature_row['market_positive_ratio'] = 0.5
                
            if all_trends:
                feature_row['market_avg_monthly_trend'] = np.mean(all_trends)
                feature_row['market_positive_trend_ratio'] = sum(1 for t in all_trends if t > 0) / len(all_trends)
            else:
                feature_row['market_avg_monthly_trend'] = 0
                feature_row['market_positive_trend_ratio'] = 0.5
                
            if all_volatilities:
                feature_row['market_avg_volatility'] = np.mean(all_volatilities)
            else:
                feature_row['market_avg_volatility'] = 0
            
            features_list.append(feature_row)
        
        features_df = pd.DataFrame(features_list)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        print(f"✅ Created {len(features_df)} feature rows with {len(features_df.columns)-1} features")
        return features_df, price_df
    
    def create_targets(self, features_df, price_df):
        """Create target labels for best ETF to hold based on future returns"""
        print("\n🎯 Creating target labels for optimal ETF selection...")
        
        targets = []
        symbols = [col for col in price_df.columns if col != 'VMOT']  # Exclude cash for now
        
        for i in range(len(features_df)):
            date = features_df.iloc[i]['date']
            
            # Find the date index in price_df
            try:
                date_idx = price_df.index.get_loc(date)
            except KeyError:
                targets.append('UPRO')  # Default
                continue
            
            # Look ahead 10 days for returns (longer horizon for better signals)
            if date_idx + 10 < len(price_df):
                future_returns = {}
                
                for symbol in symbols:
                    current_price = price_df[symbol].iloc[date_idx]
                    future_price = price_df[symbol].iloc[date_idx + 10]
                    
                    if pd.notna(current_price) and pd.notna(future_price) and current_price > 0:
                        future_returns[symbol] = (future_price / current_price) - 1
                    else:
                        future_returns[symbol] = -0.1  # Penalty for missing data
                
                # Enhanced target selection logic
                if future_returns:
                    # Get monthly trend for current date
                    current_trends = {}
                    for symbol in symbols:
                        trend_col = f'{symbol}_monthly_trend'
                        trend_value = features_df.iloc[i].get(trend_col, 0)
                        current_trends[symbol] = trend_value if pd.notna(trend_value) else 0
                    
                    # Combine future returns with monthly trend
                    combined_scores = {}
                    for symbol in symbols:
                        future_ret = future_returns.get(symbol, -0.1)
                        monthly_trend = current_trends.get(symbol, 0)
                        
                        # Weight: 70% future return, 30% monthly trend
                        combined_scores[symbol] = 0.7 * future_ret + 0.3 * monthly_trend
                    
                    # Choose the ETF with highest combined score
                    best_etf = max(combined_scores.keys(), key=lambda x: combined_scores[x])
                    
                    # Only switch if the best option is significantly better (>2% advantage)
                    if combined_scores[best_etf] > max(combined_scores[s] for s in combined_scores if s != best_etf) + 0.02:
                        targets.append(best_etf)
                    else:
                        # If no clear winner, choose based on monthly trend alone
                        best_trend_etf = max(current_trends.keys(), key=lambda x: current_trends[x])
                        targets.append(best_trend_etf)
                else:
                    targets.append('UPRO')  # Default
            else:
                targets.append('UPRO')  # Default for end of data
        
        features_df['target'] = targets
        
        # Show target distribution
        target_counts = pd.Series(targets).value_counts()
        print(f"📊 Target distribution:")
        for symbol, count in target_counts.items():
            percentage = (count / len(targets)) * 100
            print(f"   {symbol}: {count:,} ({percentage:.1f}%)")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Prepare data for ML training with proper encoding"""
        print("\n🔧 Preparing training data...")
        
        # Remove date column for training
        feature_cols = [col for col in features_df.columns if col not in ['date', 'target']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Encode target labels for ML models
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Training data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"🎯 Target classes: {sorted(y.unique())}")
        print(f"🔢 Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X_scaled, y_encoded, X, y
    
    def train_models(self, X_scaled, y_encoded, X_original, y_original):
        """Train XGBoost, LightGBM, and Random Forest models"""
        print("\n🤖 Training ML models for ETF selection...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 1. XGBoost
        print("🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softprob'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        
        # 2. LightGBM
        print("⚡ Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            objective='multiclass'
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        
        # 3. Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
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
        
        # Feature importance for best model
        best_model_name = max([('xgboost', xgb_accuracy), ('lightgbm', lgb_accuracy), ('random_forest', rf_accuracy)], key=lambda x: x[1])[0]
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔍 Top 10 Features ({best_model_name.upper()}):")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"   {i+1:2d}. {feature:<35} {importance:.4f}")
        
        return {
            'xgboost_accuracy': xgb_accuracy,
            'lightgbm_accuracy': lgb_accuracy,
            'random_forest_accuracy': rf_accuracy
        }
    
    def backtest_strategy(self, features_df, price_df, model_name):
        """Backtest the single ETF switching strategy"""
        print(f"\n📈 Backtesting {model_name.upper()} single ETF switching strategy...")
        
        model = self.models[model_name]
        initial_capital = 10000
        portfolio_value = initial_capital
        current_etf = None  # Start with no position
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
            
            # Make prediction
            if i < len(X_scaled):
                prediction_encoded = model.predict(X_scaled[i:i+1])[0]
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Get prediction confidence (probability)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled[i:i+1])[0]
                    max_prob = np.max(probabilities)
                    confidence = max_prob
                else:
                    confidence = 0.6  # Default confidence
                
                # Only make trades with high confidence (>60%) and profitable potential
                should_trade = False
                if current_etf is None:
                    # First purchase
                    should_trade = confidence > 0.6
                elif prediction != current_etf:
                    # Switching decision: require higher confidence for switches
                    should_trade = confidence > 0.65
                
                # Execute trade if conditions are met
                if should_trade and prediction in price_df.columns:
                    try:
                        new_price = price_df.loc[date, prediction]
                        if pd.notna(new_price) and new_price > 0:
                            # Sell current position if any
                            if current_etf and shares_owned > 0:
                                sell_price = price_df.loc[date, current_etf]
                                if pd.notna(sell_price) and sell_price > 0:
                                    cash = shares_owned * sell_price - transaction_cost
                                    shares_owned = 0
                            
                            # Buy new position
                            if cash > transaction_cost + 100:  # Minimum trade size
                                shares_owned = (cash - transaction_cost) / new_price
                                cash = 0
                                
                                trades.append({
                                    'date': date.strftime('%Y-%m-%d'),
                                    'action': f'Switch to {prediction}' if current_etf else f'Initial buy {prediction}',
                                    'from_etf': current_etf,
                                    'to_etf': prediction,
                                    'price': new_price,
                                    'shares': shares_owned,
                                    'confidence': confidence,
                                    'cost': transaction_cost
                                })
                                
                                current_etf = prediction
                    except (KeyError, IndexError):
                        pass  # Skip if price not available
            
            # Calculate current portfolio value
            if current_etf and shares_owned > 0:
                try:
                    current_price = price_df.loc[date, current_etf]
                    if pd.notna(current_price) and current_price > 0:
                        portfolio_value = shares_owned * current_price + cash
                    else:
                        portfolio_value = cash  # Can't value position
                except (KeyError, IndexError):
                    portfolio_value = cash
            else:
                portfolio_value = cash
            
            # Record portfolio history
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_etf': current_etf,
                'shares_owned': shares_owned,
                'cash': cash
            })
        
        # Final portfolio value
        if current_etf and shares_owned > 0:
            try:
                final_price = price_df[current_etf].iloc[-1]
                if pd.notna(final_price):
                    final_portfolio_value = shares_owned * final_price + cash
                else:
                    final_portfolio_value = cash
            except:
                final_portfolio_value = cash
        else:
            final_portfolio_value = cash
        
        total_return = (final_portfolio_value / initial_capital) - 1
        num_trades = len(trades)
        total_transaction_costs = num_trades * transaction_cost
        
        # Calculate some performance metrics
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (total_return * 252 / len(returns) - 0.02) / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"✅ {model_name.upper()} Results:")
        print(f"   💰 Final Value: ${final_portfolio_value:,.2f}")
        print(f"   📈 Total Return: {total_return*100:+.1f}%")
        print(f"   🔄 Number of Trades: {num_trades}")
        print(f"   💸 Transaction Costs: ${total_transaction_costs}")
        print(f"   📊 Volatility: {volatility*100:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {max_drawdown*100:.1f}%")
        
        return {
            'total_return': total_return,
            'final_value': final_portfolio_value,
            'num_trades': num_trades,
            'total_transaction_costs': total_transaction_costs,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'portfolio_history': portfolio_history
        }
    
    def run_complete_analysis(self):
        """Run complete ML-based single ETF switching analysis"""
        print("🚀 ML-BASED SINGLE ETF SWITCHING STRATEGY")
        print("=" * 60)
        print("Strategy: Determine optimal ETF and timing simultaneously")
        print("Models: XGBoost, LightGBM, Random Forest")
        print("Key Feature: Monthly_Trend + Technical Indicators")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_data()
        features_df, price_df = self.create_features(df)
        features_df = self.create_targets(features_df, price_df)
        
        # Prepare training data
        X_scaled, y_encoded, X_original, y_original = self.prepare_training_data(features_df)
        
        # Train models
        model_performance = self.train_models(X_scaled, y_encoded, X_original, y_original)
        
        # Backtest all strategies
        backtest_results = {}
        for model_name in ['xgboost', 'lightgbm', 'random_forest']:
            backtest_results[model_name] = self.backtest_strategy(features_df, price_df, model_name)
        
        # Add buy-and-hold benchmark (UPRO)
        print(f"\n📊 Calculating UPRO Buy-and-Hold Benchmark...")
        upro_start = price_df['UPRO'].iloc[0]
        upro_end = price_df['UPRO'].iloc[-1]
        upro_return = (upro_end / upro_start) - 1 if upro_start > 0 else 0
        
        # Save results
        results = {
            'strategy_info': {
                'name': 'ML-Based Single ETF Switching Strategy',
                'description': 'Uses ML to determine both ETF selection and timing',
                'models_used': ['XGBoost', 'LightGBM', 'Random Forest'],
                'key_features': ['monthly_trend', 'technical_indicators', 'momentum', 'volatility'],
                'date_range': f"{features_df['date'].min()} to {features_df['date'].max()}",
                'total_samples': len(features_df),
                'features_count': len(self.feature_columns)
            },
            'model_performance': model_performance,
            'backtests': backtest_results,
            'benchmark': {
                'upro_buy_hold_return': upro_return,
                'upro_start_price': upro_start,
                'upro_end_price': upro_end
            }
        }
        
        with open('single_etf_ml_switching_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to 'single_etf_ml_switching_results.json'")
        
        # Print comprehensive summary
        print(f"\n🎯 ML SINGLE ETF SWITCHING STRATEGY SUMMARY:")
        print("=" * 60)
        print(f"{'Model':<15} {'Return':<10} {'Trades':<8} {'Sharpe':<8} {'Max DD':<8}")
        print("-" * 60)
        
        for model_name, result in backtest_results.items():
            return_pct = result['total_return'] * 100
            sharpe = result['sharpe_ratio']
            max_dd = result['max_drawdown'] * 100
            trades = result['num_trades']
            print(f"{model_name.upper():<15} {return_pct:+6.1f}%   {trades:<8} {sharpe:+6.2f}  {max_dd:6.1f}%")
        
        print(f"{'UPRO B&H':<15} {upro_return*100:+6.1f}%   {'0':<8} {'N/A':<6}  {'N/A':<6}")
        print("=" * 60)
        
        # Best model
        best_model = max(backtest_results.items(), key=lambda x: x[1]['total_return'])
        print(f"\n🏆 Best Performing Model: {best_model[0].upper()}")
        print(f"   📈 Return: {best_model[1]['total_return']*100:+.1f}%")
        print(f"   🔄 Trades: {best_model[1]['num_trades']}")
        print(f"   ⚡ Sharpe: {best_model[1]['sharpe_ratio']:.2f}")
        
        return results

if __name__ == "__main__":
    strategy = SingleETFMLSwitchingStrategy()
    results = strategy.run_complete_analysis()