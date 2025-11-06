#!/usr/bin/env python3
"""
Debug script to check UPRO benchmark calculation
"""

import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('etf_data.db')

# Get UPRO data from 2020-01-01 onwards
query = """
SELECT date, close
FROM prices 
WHERE symbol = 'UPRO' AND date >= '2020-01-01'
ORDER BY date
"""

upro_data = pd.read_sql_query(query, conn)
conn.close()

upro_data['date'] = pd.to_datetime(upro_data['date'])

print("🔍 UPRO Benchmark Analysis")
print(f"📅 Date range: {upro_data['date'].min()} to {upro_data['date'].max()}")
print(f"📊 Total records: {len(upro_data)}")

# Find the data around 2020-07-01 (when strategy starts)
strategy_start_date = pd.to_datetime('2020-07-01')
start_data = upro_data[upro_data['date'] >= strategy_start_date].iloc[0]

print(f"\n📈 Strategy Start (around 2020-07-01):")
print(f"   Date: {start_data['date']}")
print(f"   UPRO Price: ${start_data['close']:.2f}")

# Get the latest data
end_data = upro_data.iloc[-1]
print(f"\n📈 Latest Data:")
print(f"   Date: {end_data['date']}")
print(f"   UPRO Price: ${end_data['close']:.2f}")

# Calculate returns
start_price = start_data['close']
end_price = end_data['close']
total_return = (end_price - start_price) / start_price

print(f"\n💰 UPRO Benchmark Calculation:")
print(f"   Start Price: ${start_price:.2f}")
print(f"   End Price: ${end_price:.2f}")
print(f"   Total Return: {total_return:.2%}")

# Calculate what $10,000 investment would be worth
initial_investment = 10000
shares_bought = initial_investment / start_price
final_value = shares_bought * end_price
strategy_return = (final_value - initial_investment) / initial_investment

print(f"\n🧮 Strategy Simulation:")
print(f"   Initial Investment: ${initial_investment:,.2f}")
print(f"   Shares Bought: {shares_bought:.2f}")
print(f"   Final Value: ${final_value:,.2f}")
print(f"   Strategy Return: {strategy_return:.2%}")

# Check if there are any data gaps or inconsistencies
print(f"\n🔍 Data Quality Check:")
print(f"   First 5 records:")
print(upro_data.head())
print(f"\n   Last 5 records:")
print(upro_data.tail())

# Check for the specific date used in benchmark calculation
benchmark_data = upro_data[upro_data['date'] >= '2020-01-01']
if len(benchmark_data) > 0:
    bench_start = benchmark_data.iloc[0]
    bench_end = benchmark_data.iloc[-1]
    bench_return = (bench_end['close'] - bench_start['close']) / bench_start['close']
    
    print(f"\n📊 Original Benchmark Calculation (from 2020-01-01):")
    print(f"   Start Date: {bench_start['date']}")
    print(f"   Start Price: ${bench_start['close']:.2f}")
    print(f"   End Date: {bench_end['date']}")
    print(f"   End Price: ${bench_end['close']:.2f}")
    print(f"   Return: {bench_return:.2%}")