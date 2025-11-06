#!/usr/bin/env python3

import sqlite3
import os

def check_database():
    db_path = 'etf_data.db'
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check UPRO data
        print("=== UPRO Data Check ===")
        cursor.execute('SELECT symbol, date FROM prices WHERE symbol = "UPRO" ORDER BY date DESC LIMIT 10')
        recent_dates = cursor.fetchall()
        print("Recent UPRO dates:")
        for row in recent_dates:
            print(f"  {row[0]}: {row[1]}")
        
        # Check date range
        cursor.execute('SELECT MIN(date), MAX(date) FROM prices WHERE symbol = "UPRO"')
        date_range = cursor.fetchone()
        print(f"UPRO date range: {date_range[0]} to {date_range[1]}")
        
        # Check specific date
        test_date = '2025-03-19'
        cursor.execute('SELECT symbol, date, close FROM prices WHERE symbol = "UPRO" AND date = ?', (test_date,))
        result = cursor.fetchone()
        if result:
            print(f"Found UPRO on {test_date}: {result}")
        else:
            print(f"No UPRO data found for {test_date}")
            
            # Check closest dates
            cursor.execute('SELECT symbol, date, close FROM prices WHERE symbol = "UPRO" AND date <= ? ORDER BY date DESC LIMIT 3', (test_date,))
            closest = cursor.fetchall()
            print("Closest dates before:")
            for row in closest:
                print(f"  {row[1]}: ${row[2]}")
        
        # Check all available symbols
        print("\n=== Available Symbols ===")
        cursor.execute('SELECT DISTINCT symbol FROM prices ORDER BY symbol')
        symbols = cursor.fetchall()
        print("Available ETF symbols:")
        for symbol in symbols:
            print(f"  {symbol[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check_database()