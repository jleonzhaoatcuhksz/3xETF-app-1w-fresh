#!/usr/bin/env python3

import json
import numpy as np
import sqlite3

def get_etf_price(symbol, date):
    """Get ETF price from database for a specific date"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Try exact date first
    cursor.execute("SELECT close FROM prices WHERE symbol = ? AND date = ?", (symbol, date))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return float(result[0])
    
    # If exact date not found, get closest previous date
    cursor.execute("""
        SELECT close FROM prices 
        WHERE symbol = ? AND date <= ? 
        ORDER BY date DESC 
        LIMIT 1
    """, (symbol, date))
    result = cursor.fetchone()
    
    conn.close()
    return float(result[0]) if result else None

def extract_test_period_results_with_upro_start():
    """Extract test period results starting with UPRO ETF worth $10,000"""
    
    # Load the original full results
    with open('single_etf_ml_switching_results.json', 'r') as f:
        data = json.load(f)
    
    test_start_date = '2024-08-20'
    initial_value = 10000
    test_results = {}
    
    print("🔄 Extracting test period results starting with UPRO ETF ($10,000)...")
    
    # Get UPRO price at test start date
    upro_start_price = get_etf_price('UPRO', test_start_date)
    if not upro_start_price:
        print(f"❌ Could not find UPRO price for {test_start_date}")
        return
    
    initial_upro_shares = initial_value / upro_start_price
    print(f"📊 Starting with {initial_upro_shares:.4f} UPRO shares at ${upro_start_price:.2f} = ${initial_value:,.2f}")
    
    # Handle ML models from backtests
    if 'backtests' in data:
        for model_name, model_data in data['backtests'].items():
            portfolio_history = model_data['portfolio_history']
            trades = model_data.get('trades', [])
            
            # Filter to test period
            test_portfolio = [entry for entry in portfolio_history if entry['date'] >= test_start_date]
            test_trades = [trade for trade in trades if trade['date'] >= test_start_date]
            
            if not test_portfolio:
                continue
            
            print(f"\n📈 Processing {model_name}...")
            
            # Create new portfolio history starting with UPRO
            new_portfolio_history = []
            current_etf = 'UPRO'
            current_shares = initial_upro_shares
            current_value = initial_value
            
            # Add initial UPRO position
            new_portfolio_history.append({
                'date': test_start_date,
                'portfolio_value': current_value,
                'shares_owned': current_shares,
                'current_etf': current_etf
            })
            
            # Process each trade
            for trade in test_trades:
                trade_date = trade['date']
                to_etf = trade['to_etf']
                
                # Get current ETF price at trade date (for selling current position)
                current_etf_price = get_etf_price(current_etf, trade_date)
                if not current_etf_price:
                    print(f"  ❌ No price for {current_etf} on {trade_date}")
                    continue
                
                # Calculate portfolio value before trade (what we get from selling current ETF)
                portfolio_value_before_trade = current_shares * current_etf_price
                
                # Get new ETF price at trade date (for buying new position)
                new_etf_price = get_etf_price(to_etf, trade_date)
                if not new_etf_price:
                    print(f"  ❌ No price for {to_etf} on {trade_date}")
                    continue
                
                # Execute the trade (sell all current ETF, buy new ETF)
                new_shares = portfolio_value_before_trade / new_etf_price
                
                # Calculate portfolio value after trade (should be same as before, minus any fees)
                portfolio_value_after_trade = new_shares * new_etf_price
                
                # Update trade record with from ETF info
                trade['from_etf'] = current_etf
                trade['from_etf_price'] = current_etf_price
                
                # Update current position
                current_etf = to_etf
                current_shares = new_shares
                
                # Add to portfolio history (use the value after trade execution)
                new_portfolio_history.append({
                    'date': trade_date,
                    'portfolio_value': portfolio_value_after_trade,
                    'shares_owned': current_shares,
                    'current_etf': current_etf
                })
                
                print(f"  📅 {trade_date}: {trade['from_etf']} → {to_etf} | Before: ${portfolio_value_before_trade:,.2f} | After: ${portfolio_value_after_trade:,.2f}")
            
            # Add daily portfolio values for all dates in test period
            # Get all unique dates from the original portfolio history
            all_dates = sorted(set([entry['date'] for entry in test_portfolio]))
            
            # Track current position through time
            current_position = {'etf': 'UPRO', 'shares': initial_upro_shares}
            
            for date in all_dates:
                # Check if there's a trade on this date
                trade_on_date = None
                for trade in test_trades:
                    if trade['date'] == date:
                        trade_on_date = trade
                        break
                
                # If there's a trade, execute it first
                if trade_on_date:
                    # Get current ETF price for selling
                    current_etf_price = get_etf_price(current_position['etf'], date)
                    portfolio_value_before = current_position['shares'] * current_etf_price
                    
                    # Get new ETF price for buying
                    new_etf_price = get_etf_price(trade_on_date['to_etf'], date)
                    new_shares = portfolio_value_before / new_etf_price
                    
                    # Update position
                    current_position = {'etf': trade_on_date['to_etf'], 'shares': new_shares}
                
                # Calculate portfolio value for this date with current position
                etf_price = get_etf_price(current_position['etf'], date)
                if etf_price:
                    portfolio_value = current_position['shares'] * etf_price
                    
                    # Only add if not already in history (from trade processing)
                    if date not in [h['date'] for h in new_portfolio_history]:
                        new_portfolio_history.append({
                            'date': date,
                            'portfolio_value': portfolio_value,
                            'shares_owned': current_position['shares'],
                            'current_etf': current_position['etf']
                        })
            
            # Sort portfolio history by date
            new_portfolio_history.sort(key=lambda x: x['date'])
            
            # Calculate final metrics
            final_value = new_portfolio_history[-1]['portfolio_value']
            total_return_pct = ((final_value - initial_value) / initial_value) * 100
            
            # Calculate performance metrics
            daily_returns = []
            for i in range(1, len(new_portfolio_history)):
                prev_value = new_portfolio_history[i-1]['portfolio_value']
                curr_value = new_portfolio_history[i]['portfolio_value']
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
            
            if daily_returns:
                daily_returns = np.array(daily_returns)
                volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
                mean_return = np.mean(daily_returns) * 252  # Annualized
                sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                peak = new_portfolio_history[0]['portfolio_value']
                max_drawdown = 0
                for entry in new_portfolio_history:
                    if entry['portfolio_value'] > peak:
                        peak = entry['portfolio_value']
                    drawdown = (peak - entry['portfolio_value']) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Map model names to display names
            model_display_names = {
                'xgboost': 'XGBoost',
                'lightgbm': 'LightGBM', 
                'random_forest': 'Random Forest'
            }
            
            test_results[model_name] = {
                'model': model_display_names.get(model_name, model_name.title()),
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return_pct': total_return_pct,
                'num_trades': len(test_trades),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'period': f"{test_start_date} to {new_portfolio_history[-1]['date']}",
                'portfolio_history': new_portfolio_history,
                'trades': test_trades
            }
            
            print(f"📊 {model_name}: {total_return_pct:+.2f}% return, {len(test_trades)} trades, Final: ${final_value:,.2f}")
    
    # Add UPRO benchmark (buy and hold)
    upro_portfolio_history = []
    
    # Get all dates in test period from one of the models
    if test_results:
        first_model = list(test_results.keys())[0]
        test_dates = [entry['date'] for entry in test_results[first_model]['portfolio_history']]
        
        for date in test_dates:
            upro_price = get_etf_price('UPRO', date)
            if upro_price:
                portfolio_value = initial_upro_shares * upro_price
                upro_portfolio_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'shares_owned': initial_upro_shares,
                    'current_etf': 'UPRO'
                })
        
        # Calculate UPRO benchmark metrics
        if upro_portfolio_history:
            final_upro_value = upro_portfolio_history[-1]['portfolio_value']
            upro_return_pct = ((final_upro_value - initial_value) / initial_value) * 100
            
            # Calculate UPRO daily returns
            upro_daily_returns = []
            for i in range(1, len(upro_portfolio_history)):
                prev_value = upro_portfolio_history[i-1]['portfolio_value']
                curr_value = upro_portfolio_history[i]['portfolio_value']
                daily_return = (curr_value - prev_value) / prev_value
                upro_daily_returns.append(daily_return)
            
            if upro_daily_returns:
                upro_daily_returns = np.array(upro_daily_returns)
                upro_volatility = np.std(upro_daily_returns) * np.sqrt(252)
                upro_mean_return = np.mean(upro_daily_returns) * 252
                upro_sharpe_ratio = upro_mean_return / upro_volatility if upro_volatility > 0 else 0
                
                # Calculate UPRO max drawdown
                upro_peak = upro_portfolio_history[0]['portfolio_value']
                upro_max_drawdown = 0
                for entry in upro_portfolio_history:
                    if entry['portfolio_value'] > upro_peak:
                        upro_peak = entry['portfolio_value']
                    drawdown = (upro_peak - entry['portfolio_value']) / upro_peak
                    upro_max_drawdown = max(upro_max_drawdown, drawdown)
            else:
                upro_volatility = 0
                upro_sharpe_ratio = 0
                upro_max_drawdown = 0
            
            test_results['upro_benchmark'] = {
                'model': 'UPRO Benchmark',
                'initial_value': initial_value,
                'final_value': final_upro_value,
                'total_return_pct': upro_return_pct,
                'sharpe_ratio': upro_sharpe_ratio,
                'max_drawdown': upro_max_drawdown,
                'volatility': upro_volatility,
                'period': f"{test_start_date} to {upro_portfolio_history[-1]['date']}",
                'portfolio_history': upro_portfolio_history
            }
            
            print(f"\n📊 UPRO Benchmark: {upro_return_pct:+.2f}% return, Final: ${final_upro_value:,.2f}")
    
    # Save the results
    output_file = 'single_etf_ml_switching_results_test_only.json'
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test period results extracted and saved to {output_file}")
    print("🚀 All models now start with UPRO ETF worth $10,000")
    
    # Print summary
    print("\n📈 SUMMARY (Test Period: 2024-08-20 to 2025-10-17):")
    print("=" * 70)
    for model_name, model_data in test_results.items():
        if model_name != 'upro_benchmark':
            print(f"{model_data['model']:15} | {model_data['total_return_pct']:+8.2f}% | {model_data['num_trades']:3d} trades | Final: ${model_data['final_value']:,.2f}")
    
    if 'upro_benchmark' in test_results:
        benchmark = test_results['upro_benchmark']
        print(f"{'UPRO Benchmark':15} | {benchmark['total_return_pct']:+8.2f}% | Buy & Hold | Final: ${benchmark['final_value']:,.2f}")

if __name__ == "__main__":
    extract_test_period_results_with_upro_start()