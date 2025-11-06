import sqlite3
import pandas as pd
import numpy as np

# Connect to database
conn = sqlite3.connect('etf_data.db')

# Get table structure
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(prices)")
columns = cursor.fetchall()
print("Database columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Get available ETF symbols
cursor.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
symbols = [row[0] for row in cursor.fetchall()]
print(f"\nAvailable ETF symbols ({len(symbols)}):")
print(symbols)

# Get date range
cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
date_range = cursor.fetchone()
print(f"\nDate range: {date_range[0]} to {date_range[1]}")

# Sample data structure
cursor.execute("SELECT * FROM prices WHERE symbol = 'SPY' ORDER BY date DESC LIMIT 5")
sample_data = cursor.fetchall()
print(f"\nSample SPY data (latest 5 records):")
for row in sample_data:
    print(row)

# Check data completeness for key ETFs
key_etfs = ['SPY', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU']
print(f"\nData completeness for key ETFs:")
for etf in key_etfs:
    cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (etf,))
    count = cursor.fetchone()[0]
    if count > 0:
        cursor.execute("SELECT MIN(date), MAX(date) FROM prices WHERE symbol = ?", (etf,))
        etf_range = cursor.fetchone()
        print(f"  {etf}: {count} records ({etf_range[0]} to {etf_range[1]})")

conn.close()