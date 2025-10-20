#!/usr/bin/env python3

import sqlite3
import os

def check_october_data():
    db_path = 'etf_data.db'
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check data around 2024-10-07
        print("=== UPRO Data Around 2024-10-07 ===")
        cursor.execute('''
            SELECT symbol, date, close 
            FROM prices 
            WHERE symbol = "UPRO" AND date BETWEEN "2024-10-01" AND "2024-10-15" 
            ORDER BY date
        ''')
        results = cursor.fetchall()
        
        if results:
            print("UPRO data found:")
            for row in results:
                print(f"  {row[1]}: ${row[2]:.2f}")
        else:
            print("No UPRO data found in October 2024")
        
        # Check exact date
        cursor.execute('SELECT symbol, date, close FROM prices WHERE symbol = "UPRO" AND date = "2024-10-07"')
        exact_result = cursor.fetchone()
        
        print(f"\nExact search for 2024-10-07:")
        if exact_result:
            print(f"  Found: {exact_result[1]} - ${exact_result[2]:.2f}")
        else:
            print("  Not found")
        
        # Check if it's a weekend (markets closed)
        import datetime
        date_obj = datetime.datetime.strptime('2024-10-07', '%Y-%m-%d')
        weekday = date_obj.strftime('%A')
        print(f"  2024-10-07 was a {weekday}")
        
        # Find closest trading days
        print("\nClosest trading days:")
        cursor.execute('''
            SELECT symbol, date, close 
            FROM prices 
            WHERE symbol = "UPRO" AND date <= "2024-10-07" 
            ORDER BY date DESC 
            LIMIT 3
        ''')
        before_results = cursor.fetchall()
        
        cursor.execute('''
            SELECT symbol, date, close 
            FROM prices 
            WHERE symbol = "UPRO" AND date >= "2024-10-07" 
            ORDER BY date ASC 
            LIMIT 3
        ''')
        after_results = cursor.fetchall()
        
        print("  Before 2024-10-07:")
        for row in before_results:
            print(f"    {row[1]}: ${row[2]:.2f}")
        
        print("  After 2024-10-07:")
        for row in after_results:
            print(f"    {row[1]}: ${row[2]:.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check_october_data()