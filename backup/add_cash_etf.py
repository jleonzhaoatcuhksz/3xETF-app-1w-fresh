#!/usr/bin/env python3

import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def add_cash_etf_to_database():
    """Add a cash ETF (money market fund) to the database"""
    
    # Connect to database
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Add VMOT (Vanguard Ultra-Short Bond ETF) as cash proxy
    # This is a very low-risk, cash-like ETF
    cash_etf = {
        'symbol': 'VMOT',
        'name': 'Vanguard Ultra-Short Bond ETF',
        'sector': 'Cash/Money Market',
        'category': 'Ultra-Short Bond'
    }
    
    print(f"Adding cash ETF: {cash_etf['symbol']} - {cash_etf['name']}")
    
    # Insert the cash ETF into etfs table
    cursor.execute('''
        INSERT OR REPLACE INTO etfs (symbol, name, sector, category) 
        VALUES (?, ?, ?, ?)
    ''', (cash_etf['symbol'], cash_etf['name'], cash_etf['sector'], cash_etf['category']))
    
    # Fetch 10 years of historical data for VMOT
    print("Fetching 10 years of historical data for VMOT...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)  # 10 years
    
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(cash_etf['symbol'])
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"❌ No data found for {cash_etf['symbol']}")
            return
        
        print(f"📊 Downloaded {len(hist)} records for {cash_etf['symbol']}")
        
        # Insert price data
        records_added = 0
        for date, row in hist.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT OR REPLACE INTO prices (date, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                cash_etf['symbol'],
                round(row['Open'], 4),
                round(row['High'], 4),
                round(row['Low'], 4),
                round(row['Close'], 4),
                int(row['Volume'])
            ))
            records_added += 1
        
        # Commit changes
        conn.commit()
        print(f"✅ Successfully added {records_added} price records for {cash_etf['symbol']}")
        
        # Show updated ETF count
        cursor.execute("SELECT COUNT(*) FROM etfs")
        etf_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (cash_etf['symbol'],))
        price_count = cursor.fetchone()[0]
        
        print(f"📈 Total ETFs in database: {etf_count}")
        print(f"📊 Price records for {cash_etf['symbol']}: {price_count}")
        
        # Show all ETFs now in database
        print("\n📋 All ETFs in database:")
        cursor.execute("SELECT symbol, name, sector FROM etfs ORDER BY symbol")
        etfs = cursor.fetchall()
        
        for i, (symbol, name, sector) in enumerate(etfs, 1):
            print(f"{i:2d}. {symbol:<6} - {name:<40} ({sector})")
            
    except Exception as e:
        print(f"❌ Error fetching data for {cash_etf['symbol']}: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    add_cash_etf_to_database()