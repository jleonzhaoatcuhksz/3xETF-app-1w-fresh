#!/usr/bin/env python3

import sqlite3
import pandas as pd

def examine_database():
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Available tables:", [table[0] for table in tables])
    
    # Examine prices table structure
    cursor.execute("PRAGMA table_info(prices)")
    columns = cursor.fetchall()
    print("\nPrices table columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Check for Monthly_Trend column
    column_names = [col[1] for col in columns]
    if 'Monthly_Trend' in column_names:
        print("\n✅ Monthly_Trend column found!")
        cursor.execute("SELECT DISTINCT Monthly_Trend FROM prices WHERE Monthly_Trend IS NOT NULL LIMIT 10")
        trends = cursor.fetchall()
        print("Sample Monthly_Trend values:", [t[0] for t in trends])
    else:
        print("\n❌ Monthly_Trend column not found in prices table")
        print("Available columns:", column_names)
    
    # Sample data
    cursor.execute("SELECT * FROM prices LIMIT 5")
    sample = cursor.fetchall()
    print("\nSample data (first 5 rows):")
    for row in sample:
        print(row)
    
    # Check date range and symbols
    cursor.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
    symbols = [s[0] for s in cursor.fetchall()]
    print(f"\nAvailable symbols ({len(symbols)}): {symbols}")
    
    cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
    date_range = cursor.fetchone()
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    cursor.execute("SELECT COUNT(*) FROM prices")
    total_records = cursor.fetchone()[0]
    print(f"Total records: {total_records:,}")
    
    conn.close()

if __name__ == "__main__":
    examine_database()