#!/usr/bin/env python3

import json
import sqlite3
import pandas as pd
from datetime import datetime

def add_from_etf_prices():
    """Add From ETF prices to the test-only trade data"""
    
    # Load the test-only results
    with open('single_etf_ml_switching_results_test_only.json', 'r') as f:
        data = json.load(f)
    
    # Connect to database
    conn = sqlite3.connect('etf_data.db')
    
    print("🔄 Adding From ETF prices to trade data...")
    
    # Process each model's trades
    for model_name, model_data in data.items():
        if model_name == 'upro_benchmark' or 'trades' not in model_data:
            continue
            
        print(f"📊 Processing {model_name}...")
        trades = model_data['trades']
        
        for trade in trades:
            if trade.get('from_etf') and trade['from_etf'] != 'CASH':
                # Get the price of the From ETF on the trade date
                query = """
                SELECT close 
                FROM prices 
                WHERE symbol = ? AND date = ?
                """
                
                cursor = conn.execute(query, (trade['from_etf'], trade['date']))
                result = cursor.fetchone()
                
                if result:
                    trade['from_etf_price'] = round(result[0], 2)
                    print(f"  📈 {trade['date']}: {trade['from_etf']} = ${trade['from_etf_price']}")
                else:
                    # Try to find the closest date
                    query_closest = """
                    SELECT close, date
                    FROM prices 
                    WHERE symbol = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 1
                    """
                    
                    cursor = conn.execute(query_closest, (trade['from_etf'], trade['date']))
                    result = cursor.fetchone()
                    
                    if result:
                        trade['from_etf_price'] = round(result[0], 2)
                        print(f"  📈 {trade['date']}: {trade['from_etf']} = ${trade['from_etf_price']} (closest date: {result[1]})")
                    else:
                        trade['from_etf_price'] = 'N/A'
                        print(f"  ❌ {trade['date']}: No price data found for {trade['from_etf']}")
            else:
                trade['from_etf_price'] = 'N/A'  # CASH or no from_etf
    
    conn.close()
    
    # Save the updated data
    with open('single_etf_ml_switching_results_test_only.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ From ETF prices added successfully!")
    print("📄 Updated file: single_etf_ml_switching_results_test_only.json")

if __name__ == "__main__":
    add_from_etf_prices()