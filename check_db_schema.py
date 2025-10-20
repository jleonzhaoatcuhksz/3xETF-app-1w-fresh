#!/usr/bin/env python3

import sqlite3

def check_database_schema():
    """Check the database schema to understand the table structure"""
    
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(prices)")
    columns = cursor.fetchall()
    
    print("📊 Database Schema for 'prices' table:")
    print("=" * 50)
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Show sample data
    print("\n📋 Sample data from prices table:")
    print("=" * 50)
    cursor.execute("SELECT * FROM prices LIMIT 5")
    rows = cursor.fetchall()
    
    # Get column names
    col_names = [description[0] for description in cursor.description]
    print("Columns:", col_names)
    
    for row in rows:
        print(row)
    
    conn.close()

if __name__ == "__main__":
    check_database_schema()