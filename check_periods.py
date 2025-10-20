#!/usr/bin/env python3

import sqlite3
import pandas as pd

# Connect to database and get date range
conn = sqlite3.connect('etf_data.db')

# Get all unique dates from 2020-01-01 onwards (same as ML strategy)
query = """
SELECT DISTINCT date 
FROM prices 
WHERE date >= '2020-01-01'
ORDER BY date
"""

dates_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate 80/20 split (same as ML strategy uses)
total_dates = len(dates_df)
split_index = int(total_dates * 0.8)

print("=== ML Strategy Training and Testing Periods ===")
print(f"Total unique training dates: {total_dates}")
print(f"Training dates (80%): {split_index}")
print(f"Testing dates (20%): {total_dates - split_index}")
print()

# Get the actual date ranges
first_date = dates_df.iloc[0]['date']
split_end_date = dates_df.iloc[split_index-1]['date'] 
test_start_date = dates_df.iloc[split_index]['date']
last_date = dates_df.iloc[-1]['date']

print(f"📊 TRAINING PERIOD: {first_date} to {split_end_date}")
print(f"🧪 TESTING PERIOD:  {test_start_date} to {last_date}")
print()

# Show some additional info
print("Additional Information:")
print(f"- Data starts from: {first_date}")
print(f"- Data ends at: {last_date}")
print(f"- Total data span: {total_dates} trading days")
print(f"- Training uses first {split_index} days")
print(f"- Testing uses last {total_dates - split_index} days")