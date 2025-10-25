#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
from datetime import datetime

def extract_test_period_results():
    """Extract only testing period results from the full ML strategy results"""
    
    # Load the full results
    try:
        with open('single_etf_ml_switching_results.json', 'r') as f:
            full_results = json.load(f)
    except FileNotFoundError:
        print("❌ Full results file not found. Please run the main strategy first.")
        return
    
    # Define test period
    test_start = '2024-08-20'
    test_end = '2025-10-17'
    
    print(f"🔍 Extracting TEST PERIOD results: {test_start} to {test_end}")
    print("=" * 60)
    
    test_results = {}
    
    # Get the backtests section which contains the actual model results
    backtests = full_results.get('backtests', {})
    
    for model_name, model_data in backtests.items():
        print(f"\n📊 Processing {model_name.upper()} model...")
        
        # Get portfolio history
        portfolio_history = model_data.get('portfolio_history', [])
        if not portfolio_history:
            print(f"   ❌ No portfolio history for {model_name}")
            continue
        
        # Filter to test period
        test_portfolio = []
        for entry in portfolio_history:
            entry_date = entry['date']
            if test_start <= entry_date <= test_end:
                test_portfolio.append(entry)
        
        if not test_portfolio:
            print(f"   ❌ No data in test period for {model_name}")
            continue
        
        # Calculate test period performance
        initial_value = test_portfolio[0]['portfolio_value']
        final_value = test_portfolio[-1]['portfolio_value']
        test_return = (final_value / initial_value) - 1
        
        # Filter trades to test period
        all_trades = model_data.get('trades', [])
        test_trades = []
        for trade in all_trades:
            trade_date = trade['date']
            if test_start <= trade_date <= test_end:
                test_trades.append(trade)
        
        # Calculate performance metrics for test period
        portfolio_values = [p['portfolio_value'] for p in test_portfolio]
        
        # Calculate returns
        returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            returns.append(daily_return)
        
        # Performance metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        avg_return = np.mean(returns) if returns else 0
        sharpe_ratio = (avg_return * 252 - 0.02) / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Store test results
        test_results[model_name] = {
            'model': model_name.upper(),
            'period': f"{test_start} to {test_end}",
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': test_return,
            'total_return_pct': test_return * 100,
            'num_trades': len(test_trades),
            'transaction_costs': len(test_trades) * 10,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_history': test_portfolio,
            'trades': test_trades,
            'days_in_period': len(test_portfolio)
        }
        
        print(f"   ✅ Test period results:")
        print(f"      📅 Period: {test_start} to {test_end} ({len(test_portfolio)} days)")
        print(f"      💰 Initial Value: ${initial_value:,.2f}")
        print(f"      💰 Final Value: ${final_value:,.2f}")
        print(f"      📈 Return: {test_return*100:+.2f}%")
        print(f"      🔄 Trades: {len(test_trades)}")
        print(f"      📊 Volatility: {volatility*100:.1f}%")
        print(f"      ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"      📉 Max Drawdown: {max_drawdown*100:.1f}%")
    
    # Calculate UPRO benchmark for test period
    print(f"\n📊 Calculating UPRO benchmark for test period...")
    try:
        import sqlite3
        conn = sqlite3.connect('etf_data.db')
        
        query = """
        SELECT date, close as price
        FROM prices 
        WHERE symbol = 'UPRO' 
        AND date >= ? AND date <= ?
        ORDER BY date
        """
        
        upro_data = pd.read_sql_query(query, conn, params=[test_start, test_end])
        conn.close()
        
        if len(upro_data) >= 2:
            upro_start_price = upro_data.iloc[0]['price']
            upro_end_price = upro_data.iloc[-1]['price']
            upro_return = (upro_end_price / upro_start_price) - 1
            
            test_results['upro_benchmark'] = {
                'model': 'UPRO_BUYHOLD',
                'period': f"{test_start} to {test_end}",
                'total_return': upro_return,
                'total_return_pct': upro_return * 100,
                'start_price': upro_start_price,
                'end_price': upro_end_price
            }
            
            print(f"   ✅ UPRO Buy-and-Hold: {upro_return*100:+.2f}%")
        else:
            print(f"   ❌ Insufficient UPRO data for test period")
        
    except Exception as e:
        print(f"   ⚠️ Could not calculate UPRO benchmark: {e}")
    
    # Save test results
    output_file = 'single_etf_ml_switching_results_test_only.json'
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Test period results saved to '{output_file}'")
    
    # Summary table
    print(f"\n🎯 ML STRATEGY PERFORMANCE - TEST PERIOD ONLY ({test_start} to {test_end}):")
    print("=" * 80)
    print("Model           Return     Trades   Sharpe   Max DD    Days")
    print("-" * 80)
    
    strategy_models = [r for k, r in test_results.items() if k != 'upro_benchmark']
    
    for results in strategy_models:
        model_display = results['model']
        return_pct = results['total_return_pct']
        trades = results['num_trades']
        sharpe = results['sharpe_ratio']
        max_dd = results['max_drawdown'] * 100
        days = results['days_in_period']
        
        print(f"{model_display:<15} {return_pct:+8.2f}%   {trades:4d}      {sharpe:+8.2f}    {max_dd:4.1f}%   {days:3d}")
    
    if 'upro_benchmark' in test_results:
        upro_result = test_results['upro_benchmark']
        print(f"UPRO B&H        {upro_result['total_return_pct']:+8.2f}%   0        N/A     N/A     N/A")
    
    print("=" * 80)
    
    # Find best performer
    if strategy_models:
        best_model = max(strategy_models, key=lambda x: x['total_return'])
        print(f"\n🏆 Best Performing Model (TEST PERIOD): {best_model['model']}")
        print(f"   📈 Return: {best_model['total_return_pct']:+.2f}%")
        print(f"   🔄 Trades: {best_model['num_trades']}")
        print(f"   ⚡ Sharpe: {best_model['sharpe_ratio']:+.2f}")
        print(f"   📉 Max DD: {best_model['max_drawdown']*100:.1f}%")
    else:
        print("\n⚠️ No strategy models found in test period")
    
    return test_results

if __name__ == "__main__":
    extract_test_period_results()