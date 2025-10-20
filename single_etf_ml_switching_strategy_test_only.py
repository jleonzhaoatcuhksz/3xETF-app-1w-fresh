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

class SingleETFMLSwitchingStrategyTestOnly:
    """
    Single ETF switching strategy - TESTING PERIOD ONLY
    Shows performance only during testing period: 2024-08-20 to 2025-10-17
    """
    
    def __init__(self, db_path='etf_data.db'):
        self.db_path = db_path
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
        # Define test period dates
        self.test_start_date = '2024-08-20'
        self.test_end_date = '2025-10-17'
        
    def load_data(self):
        """Load ETF data with monthly_trend column"""
        conn = sqlite3.connect(self.db_path)
        
        # Load price data including monthly_trend
        query = """
        SELECT date, symbol, close, volume, monthly_trend, sma_5d,
               open, high, low, adj_close
        FROM prices 
        WHERE date >= '2020-01-01'  -- Use all data for training
        ORDER BY date, symbol
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 Loaded {len(df)} records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Get unique symbols
        symbols = sorted(df['symbol'].unique())
        print(f"🎯 Symbols: {symbols}")
        
        # Check monthly_trend availability
        trend_count = df['monthly_trend'].notna().sum()
        print(f"📈 Monthly trend data available: {trend_count} records")
        
        return df
    
    def create_features(self, data):
        """Create comprehensive ML features for ETF selection"""
        print("\n🔧 Creating ML features...")
        
        # Pivot data for easier feature creation
        price_df = data.pivot(index='date', columns='symbol', values='close')
        volume_df = data.pivot(index='date', columns='symbol', values='volume')
        trend_df = data.pivot(index='date', columns='symbol', values='monthly_trend')
        
        # Create returns
        returns_df = price_df.pct_change()
        
        features_list = []
        symbols = list(price_df.columns)
        
        # Skip first 30 days to have enough history for features
        for i in range(30, len(price_df)):
            date = price_df.index[i]
            feature_row = {'date': date}
            
            # For each ETF, create comprehensive features
            for symbol in symbols:
                prices = price_df[symbol].iloc[:i+1]
                returns = returns_df[symbol].iloc[:i+1]
                volumes = volume_df[symbol].iloc[:i+1]
                
                # Skip if not enough data
                if len(prices) < 30 or prices.iloc[-1] <= 0:
                    continue
                
                # Basic price features
                current_price = prices.iloc[-1]
                feature_row[f'{symbol}_price'] = current_price
                
                # Monthly trend (key feature)
                if symbol in trend_df.columns:
                    feature_row[f'{symbol}_monthly_trend'] = trend_df[symbol].iloc[i]
                else:
                    feature_row[f'{symbol}_monthly_trend'] = 0
                
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
        symbols = [col for col in price_df.columns if col in features_df.columns]
        
        for i in range(len(features_df) - 1):  # -1 because we need next day data
            current_date = features_df.iloc[i]['date']
            next_date = features_df.iloc[i + 1]['date']
            
            # Calculate next day returns for all ETFs
            etf_returns = {}
            for symbol in symbols:
                try:
                    current_price = price_df.loc[current_date, symbol]
                    next_price = price_df.loc[next_date, symbol]
                    
                    if pd.notna(current_price) and pd.notna(next_price) and current_price > 0:
                        daily_return = (next_price / current_price) - 1
                        etf_returns[symbol] = daily_return
                except (KeyError, IndexError):
                    continue
            
            # Find the best performing ETF for next day
            if etf_returns:
                best_etf = max(etf_returns.keys(), key=lambda x: etf_returns[x])
                targets.append(best_etf)
            else:
                targets.append('CASH')  # Default if no valid returns
        
        # Add one more target for the last row (use previous best)
        if targets:
            targets.append(targets[-1])
        else:
            targets.append('CASH')
        
        features_df['target'] = targets
        
        # Show target distribution
        target_counts = pd.Series(targets).value_counts()
        print("📊 Target distribution:")
        for etf, count in target_counts.head(15).items():
            pct = (count / len(targets)) * 100
            print(f"   {etf}: {count} ({pct:.1f}%)")
        
        return features_df
    
    def train_models(self, features_df):
        """Train ML models on full dataset, but only evaluate on test period"""
        print("\n🔧 Preparing training data...")
        
        # Prepare features and targets
        feature_cols = [col for col in features_df.columns if col not in ['date', 'target']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        
        # Remove rows where target is 'CASH'
        valid_mask = y != 'CASH'
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"✅ Training data prepared: {len(X)} samples, {len(X.columns)} features")
        print(f"🎯 Target classes: {sorted(y.unique())}")
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        print(f"🔢 Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_columns = feature_cols
        
        # Split data (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n🤖 Training ML models for ETF selection...")
        
        # Train XGBoost
        print("🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        self.models['XGBOOST'] = xgb_model
        
        # Train LightGBM
        print("⚡ Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        self.models['LIGHTGBM'] = lgb_model
        
        # Train Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        self.models['RANDOM_FOREST'] = rf_model
        
        print(f"\n📊 Model Training Results:")
        print(f"🚀 XGBoost Accuracy:     {xgb_accuracy:.3f} ({xgb_accuracy*100:.1f}%)")
        print(f"⚡ LightGBM Accuracy:    {lgb_accuracy:.3f} ({lgb_accuracy*100:.1f}%)")
        print(f"🌳 Random Forest Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        
        # Show feature importance for best model
        best_model_name = 'LIGHTGBM' if lgb_accuracy >= max(xgb_accuracy, rf_accuracy) else ('XGBOOST' if xgb_accuracy >= rf_accuracy else 'RANDOM_FOREST')
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = list(zip(feature_cols, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔍 Top 10 Features ({best_model_name}):")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"   {i+1:2d}. {feature:<35} {importance:10.4f}")
    
    def backtest_strategy(self, model_name, features_df, price_df):
        """Backtest strategy - TESTING PERIOD ONLY"""
        print(f"\n📈 Backtesting {model_name} single ETF switching strategy (TEST PERIOD ONLY)...")
        
        model = self.models[model_name]
        
        # Filter to testing period only
        test_features = features_df[
            (features_df['date'] >= self.test_start_date) & 
            (features_df['date'] <= self.test_end_date)
        ].copy()
        
        if len(test_features) == 0:
            print(f"❌ No data in testing period for {model_name}")
            return None
        
        print(f"📅 Testing period: {test_features['date'].min().strftime('%Y-%m-%d')} to {test_features['date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 Testing days: {len(test_features)}")
        
        # Initialize portfolio
        initial_capital = 10000
        cash = initial_capital
        shares_owned = 0
        current_etf = None
        portfolio_history = []
        trades = []
        transaction_cost = 10
        
        # Prepare features for prediction
        feature_cols = [col for col in test_features.columns if col not in ['date', 'target']]
        X = test_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Run backtest on testing period only
        for i, (_, row) in enumerate(test_features.iterrows()):
            date = row['date']
            
            # Make prediction
            if len(X_scaled) > i:
                prediction_encoded = model.predict(X_scaled[i:i+1])[0]
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Get prediction confidence
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled[i:i+1])[0]
                    confidence = np.max(probabilities)
                else:
                    confidence = 0.6
                
                # Trading logic
                should_trade = False
                if current_etf is None:
                    should_trade = confidence > 0.6
                elif prediction != current_etf:
                    should_trade = confidence > 0.65
                
                # Execute trade
                if should_trade and prediction in price_df.columns:
                    try:
                        new_price = price_df.loc[date, prediction]
                        if pd.notna(new_price) and new_price > 0:
                            # Sell current position
                            if current_etf and shares_owned > 0:
                                sell_price = price_df.loc[date, current_etf]
                                if pd.notna(sell_price) and sell_price > 0:
                                    cash = shares_owned * sell_price - transaction_cost
                                    shares_owned = 0
                            
                            # Buy new position
                            if cash > transaction_cost + 100:
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
                        pass
            
            # Calculate portfolio value
            if current_etf and shares_owned > 0:
                try:
                    current_price = price_df.loc[date, current_etf]
                    if pd.notna(current_price) and current_price > 0:
                        portfolio_value = shares_owned * current_price + cash
                    else:
                        portfolio_value = cash
                except (KeyError, IndexError):
                    portfolio_value = cash
            else:
                portfolio_value = cash
            
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_etf': current_etf,
                'shares_owned': shares_owned,
                'cash': cash
            })
        
        # Calculate final results
        if current_etf and shares_owned > 0:
            try:
                final_date = test_features['date'].iloc[-1]
                final_price = price_df.loc[final_date, current_etf]
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
        total_costs = num_trades * transaction_cost
        
        # Performance metrics
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (total_return * 252 / len(returns) - 0.02) / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'model': model_name,
            'period': f"{self.test_start_date} to {self.test_end_date}",
            'initial_capital': initial_capital,
            'final_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': num_trades,
            'transaction_costs': total_costs,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_history': portfolio_history,
            'trades': trades
        }
        
        print(f"✅ {model_name} Results (TEST PERIOD ONLY):")
        print(f"   💰 Final Value: ${final_portfolio_value:,.2f}")
        print(f"   📈 Total Return: {total_return*100:+.1f}%")
        print(f"   🔄 Number of Trades: {num_trades}")
        print(f"   💸 Transaction Costs: ${total_costs}")
        print(f"   📊 Volatility: {volatility*100:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {max_drawdown*100:.1f}%")
        
        return results

    def run_strategy(self):
        """Run the complete ML strategy - TEST PERIOD EVALUATION ONLY"""
        print("🚀 ML-BASED SINGLE ETF SWITCHING STRATEGY - TEST PERIOD ONLY")
        print("=" * 70)
        print("Strategy: ML models trained on full data, evaluated on test period only")
        print("Test Period: 2024-08-20 to 2025-10-17")
        print("Models: XGBoost, LightGBM, Random Forest")
        print("=" * 70)
        
        # Load and prepare data
        data = self.load_data()
        features_df, price_df = self.create_features(data)
        features_df = self.create_targets(features_df, price_df)
        
        # Train models on full dataset
        self.train_models(features_df)
        
        # Backtest each model on TEST PERIOD ONLY
        all_results = {}
        for model_name in self.models.keys():
            results = self.backtest_strategy(model_name, features_df, price_df)
            if results:
                all_results[model_name.lower()] = results
        
        # Calculate benchmark (UPRO buy-and-hold for test period)
        print(f"\n📊 Calculating UPRO Buy-and-Hold Benchmark (TEST PERIOD)...")
        try:
            test_start = pd.to_datetime(self.test_start_date)
            test_end = pd.to_datetime(self.test_end_date)
            
            if 'UPRO' in price_df.columns:
                upro_start_price = price_df.loc[price_df.index >= test_start, 'UPRO'].iloc[0]
                upro_end_price = price_df.loc[price_df.index <= test_end, 'UPRO'].iloc[-1]
                
                if pd.notna(upro_start_price) and pd.notna(upro_end_price) and upro_start_price > 0:
                    upro_return = (upro_end_price / upro_start_price) - 1
                    print(f"📈 UPRO Buy-and-Hold (TEST PERIOD): {upro_return*100:+.1f}%")
                    
                    all_results['upro_benchmark'] = {
                        'model': 'UPRO_BUYHOLD',
                        'period': f"{self.test_start_date} to {self.test_end_date}",
                        'total_return': upro_return,
                        'total_return_pct': upro_return * 100
                    }
        except Exception as e:
            print(f"⚠️ Could not calculate UPRO benchmark: {e}")
        
        # Save results
        output_file = 'single_etf_ml_switching_results_test_only.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to '{output_file}'")
        
        # Summary
        print(f"\n🎯 ML SINGLE ETF SWITCHING STRATEGY SUMMARY (TEST PERIOD ONLY):")
        print("=" * 70)
        print("Model           Return     Trades   Sharpe   Max DD")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            if model_name != 'upro_benchmark':
                model_display = results['model']
                return_pct = results['total_return_pct']
                trades = results['num_trades']
                sharpe = results['sharpe_ratio']
                max_dd = results['max_drawdown'] * 100
                
                print(f"{model_display:<15} {return_pct:+8.1f}%   {trades:4d}      {sharpe:+8.2f}    {max_dd:4.1f}%")
        
        if 'upro_benchmark' in all_results:
            upro_result = all_results['upro_benchmark']
            print(f"UPRO B&H        {upro_result['total_return_pct']:+8.1f}%   0        N/A     N/A")
        
        print("=" * 70)
        
        # Find best model
        if all_results:
            best_model = max([r for k, r in all_results.items() if k != 'upro_benchmark'], 
                           key=lambda x: x['total_return'])
            print(f"\n🏆 Best Performing Model (TEST PERIOD): {best_model['model']}")
            print(f"   📈 Return: {best_model['total_return_pct']:+.1f}%")
            print(f"   🔄 Trades: {best_model['num_trades']}")
            print(f"   ⚡ Sharpe: {best_model['sharpe_ratio']:+.2f}")
        
        return all_results

if __name__ == "__main__":
    strategy = SingleETFMLSwitchingStrategyTestOnly()
    results = strategy.run_strategy()