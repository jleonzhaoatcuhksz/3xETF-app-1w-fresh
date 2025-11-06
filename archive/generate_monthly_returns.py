#!/usr/bin/env python3

import json
import sqlite3
from pathlib import Path

def generate_monthly_returns_data():
    """Generate monthly returns data for each trade"""
    
    # Load the strategy results
    with open('single_etf_ml_switching_results.json', 'r') as f:
        data = json.load(f)
    
    # Connect to database
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    monthly_returns = {}
    
    # Process each model's portfolio history
    for model_name, backtest in data['backtests'].items():
        portfolio_history = backtest['portfolio_history']
        model_monthly_returns = []
        
        # Track trades (position changes)
        for i, entry in enumerate(portfolio_history):
            # Check if this is a trade (first entry or ETF change)
            if i == 0 or entry['current_etf'] != portfolio_history[i-1]['current_etf']:
                etf = entry['current_etf']
                date = entry['date']
                
                # Get monthly return from database
                cursor.execute('''
                    SELECT monthly_trend 
                    FROM prices 
                    WHERE symbol = ? AND date = ?
                ''', (etf, date))
                
                result = cursor.fetchone()
                monthly_return = result[0] if result and result[0] is not None else 0
                
                # Get From ETF price if this is a switch (not initial trade)
                from_etf_price = None
                if i > 0:  # Not the initial trade
                    from_etf = portfolio_history[i-1]['current_etf']
                    cursor.execute('''
                        SELECT close 
                        FROM prices 
                        WHERE symbol = ? AND date = ?
                    ''', (from_etf, date))
                    
                    from_result = cursor.fetchone()
                    from_etf_price = from_result[0] if from_result else None
                
                model_monthly_returns.append({
                    'date': date,
                    'etf': etf,
                    'monthly_return': monthly_return,
                    'from_etf_price': from_etf_price
                })
        
        monthly_returns[model_name] = model_monthly_returns
    
    # Save to JSON file
    with open('monthly_returns_data.json', 'w') as f:
        json.dump(monthly_returns, f, indent=2)
    
    conn.close()
    print(f"✅ Generated monthly returns data with {len(monthly_returns)} models")
    for model, returns in monthly_returns.items():
        print(f"   {model}: {len(returns)} trades")

if __name__ == "__main__":
    generate_monthly_returns_data()