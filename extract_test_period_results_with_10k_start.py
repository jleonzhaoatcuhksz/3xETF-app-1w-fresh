#!/usr/bin/env python3

import json
import numpy as np

def extract_test_period_results_with_10k_start():
    """Extract test period results and reset starting value to $10,000"""
    
    # Load the original full results
    with open('single_etf_ml_switching_results.json', 'r') as f:
        data = json.load(f)
    
    test_start_date = '2024-08-20'
    test_results = {}
    
    print("🔄 Extracting test period results with $10,000 starting value...")
    
    for model_name, model_data in data.items():
        if model_name == 'upro_benchmark':
            # Handle benchmark separately
            benchmark_data = model_data.copy()
            
            # Find test period portfolio history
            portfolio_history = benchmark_data.get('portfolio_history', [])
            test_portfolio = [entry for entry in portfolio_history if entry['date'] >= test_start_date]
            
            if test_portfolio:
                # Reset to $10,000 start and recalculate
                original_start_value = test_portfolio[0]['portfolio_value']
                scale_factor = 10000 / original_start_value
                
                # Scale all portfolio values
                for entry in test_portfolio:
                    entry['portfolio_value'] = entry['portfolio_value'] * scale_factor
                    if 'shares_owned' in entry and entry['shares_owned'] > 0:
                        entry['shares_owned'] = entry['shares_owned'] * scale_factor
                
                # Recalculate metrics
                initial_value = 10000
                final_value = test_portfolio[-1]['portfolio_value']
                total_return_pct = ((final_value - initial_value) / initial_value) * 100
                
                # Calculate daily returns for Sharpe ratio and volatility
                daily_returns = []
                for i in range(1, len(test_portfolio)):
                    prev_value = test_portfolio[i-1]['portfolio_value']
                    curr_value = test_portfolio[i]['portfolio_value']
                    daily_return = (curr_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
                
                if daily_returns:
                    daily_returns = np.array(daily_returns)
                    volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
                    mean_return = np.mean(daily_returns) * 252  # Annualized
                    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                    
                    # Calculate max drawdown
                    peak = test_portfolio[0]['portfolio_value']
                    max_drawdown = 0
                    for entry in test_portfolio:
                        if entry['portfolio_value'] > peak:
                            peak = entry['portfolio_value']
                        drawdown = (peak - entry['portfolio_value']) / peak
                        max_drawdown = max(max_drawdown, drawdown)
                else:
                    volatility = 0
                    sharpe_ratio = 0
                    max_drawdown = 0
                
                test_results[model_name] = {
                    'model': 'UPRO Benchmark',
                    'initial_value': initial_value,
                    'final_value': final_value,
                    'total_return_pct': total_return_pct,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'period': f"{test_start_date} to {test_portfolio[-1]['date']}",
                    'portfolio_history': test_portfolio
                }
                
                print(f"📊 {model_name}: {total_return_pct:.2f}% return")
            
            continue
        
        # Handle ML models
        portfolio_history = model_data['portfolio_history']
        trades = model_data.get('trades', [])
        
        # Filter to test period
        test_portfolio = [entry for entry in portfolio_history if entry['date'] >= test_start_date]
        test_trades = [trade for trade in trades if trade['date'] >= test_start_date]
        
        if not test_portfolio:
            continue
        
        # Reset starting value to $10,000 and recalculate everything
        original_start_value = test_portfolio[0]['portfolio_value']
        scale_factor = 10000 / original_start_value
        
        print(f"📈 {model_name}: Original start: ${original_start_value:,.2f}, Scale factor: {scale_factor:.6f}")
        
        # Scale portfolio history
        for entry in test_portfolio:
            entry['portfolio_value'] = entry['portfolio_value'] * scale_factor
            if 'shares_owned' in entry and entry['shares_owned'] > 0:
                entry['shares_owned'] = entry['shares_owned'] * scale_factor
        
        # Recalculate metrics
        initial_value = 10000
        final_value = test_portfolio[-1]['portfolio_value']
        total_return_pct = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate performance metrics
        daily_returns = []
        for i in range(1, len(test_portfolio)):
            prev_value = test_portfolio[i-1]['portfolio_value']
            curr_value = test_portfolio[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        if daily_returns:
            daily_returns = np.array(daily_returns)
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            mean_return = np.mean(daily_returns) * 252  # Annualized
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            peak = test_portfolio[0]['portfolio_value']
            max_drawdown = 0
            for entry in test_portfolio:
                if entry['portfolio_value'] > peak:
                    peak = entry['portfolio_value']
                drawdown = (peak - entry['portfolio_value']) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        test_results[model_name] = {
            'model': model_data['model'],
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'num_trades': len(test_trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'period': f"{test_start_date} to {test_portfolio[-1]['date']}",
            'portfolio_history': test_portfolio,
            'trades': test_trades
        }
        
        print(f"📊 {model_name}: {total_return_pct:.2f}% return, {len(test_trades)} trades")
    
    # Save the results
    output_file = 'single_etf_ml_switching_results_test_only.json'
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test period results extracted and saved to {output_file}")
    print("💰 All models now start with $10,000 at the beginning of test period")
    
    # Print summary
    print("\n📈 SUMMARY (Test Period: 2024-08-20 to 2025-10-17):")
    print("=" * 60)
    for model_name, model_data in test_results.items():
        if model_name != 'upro_benchmark':
            print(f"{model_data['model']:15} | {model_data['total_return_pct']:+8.2f}% | {model_data['num_trades']:3d} trades | Final: ${model_data['final_value']:,.0f}")
    
    if 'upro_benchmark' in test_results:
        benchmark = test_results['upro_benchmark']
        print(f"{'UPRO Benchmark':15} | {benchmark['total_return_pct']:+8.2f}% | Buy & Hold | Final: ${benchmark['final_value']:,.0f}")

if __name__ == "__main__":
    extract_test_period_results_with_10k_start()