#!/usr/bin/env python3
"""
Simple test to verify UPRO buy-and-hold calculation
"""

import sqlite3
import pandas as pd

# Connect to database  
conn = sqlite3.connect('etf_data.db')

# Get monthly dates like the strategy does
query = """
SELECT date, close
FROM prices 
WHERE symbol = 'UPRO' AND date >= '2020-01-01'
ORDER BY date
"""

upro_data = pd.read_sql_query(query, conn)
conn.close()

upro_data['date'] = pd.to_datetime(upro_data['date'])

# Get first trading day of each month (like the strategy)
upro_data['year_month'] = upro_data['date'].dt.to_period('M')
monthly_dates = upro_data.groupby('year_month')['date'].min().values

print("📅 Monthly rebalancing dates (first 10):")
for i, date in enumerate(monthly_dates[:10]):
    price_data = upro_data[upro_data['date'] == date]
    if len(price_data) > 0:
        price = price_data.iloc[0]['close']
        print(f"   {pd.to_datetime(date).strftime('%Y-%m-%d')}: ${price:.2f}")

# Skip first 6 months like the strategy
strategy_dates = monthly_dates[6:]
print(f"\n🚀 Strategy starts at month 7: {pd.to_datetime(strategy_dates[0]).strftime('%Y-%m-%d')}")

# Get start and end data for strategy period
start_date = strategy_dates[0]
end_date = strategy_dates[-1]

start_data = upro_data[upro_data['date'] == start_date]
end_data = upro_data[upro_data['date'] == end_date]

if len(start_data) > 0 and len(end_data) > 0:
    start_price = start_data.iloc[0]['close']
    end_price = end_data.iloc[0]['close']
    
    # Simulate the exact strategy calculation
    initial_capital = 10000
    shares_owned = initial_capital / start_price
    final_value = shares_owned * end_price
    total_return = (final_value - initial_capital) / initial_capital
    
    print(f"\n💰 Exact Strategy Simulation:")
    print(f"   Start Date: {pd.to_datetime(start_date).strftime('%Y-%m-%d')}")
    print(f"   Start Price: ${start_price:.2f}")
    print(f"   End Date: {pd.to_datetime(end_date).strftime('%Y-%m-%d')}")
    print(f"   End Price: ${end_price:.2f}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Shares Bought: {shares_owned:.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    
    print(f"\n🔍 Comparison with Strategy Results:")
    print(f"   Strategy reported: $51,486.29 (414.86%)")
    print(f"   Should be: ${final_value:,.2f} ({total_return:.2%})")
    print(f"   Difference: ${51486.29 - final_value:,.2f}")