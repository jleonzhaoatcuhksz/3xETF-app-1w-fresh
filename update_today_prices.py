#!/usr/bin/env python3
"""
Script to update ETF prices from Yahoo Finance for today's data
"""

import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def get_etf_list():
    """Get list of all ETFs from the database"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT symbol, name FROM etfs ORDER BY symbol')
    etfs = cursor.fetchall()
    conn.close()
    return etfs

def get_today_price_data(symbol):
    """Fetch today's price data for a given ETF symbol"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get data for today specifically
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Try to get today's data with a 1-day range
        hist = ticker.history(period='1d')
        
        if hist.empty:
            print(f"❌ No data found for {symbol} today")
            return None
        
        # Get today's data
        latest_data = hist.iloc[-1]
        data_date = hist.index[-1].strftime('%Y-%m-%d')
        
        # Check if we got today's data
        if data_date != today:
            print(f"⚠️ Got data for {data_date} instead of {today} for {symbol}")
        
        # Get 5-day SMA by fetching last 5 days
        hist_5d = ticker.history(period='5d')
        sma_5d = float(hist_5d['Close'].mean()) if len(hist_5d) >= 5 else None
        
        return {
            'symbol': symbol,
            'date': data_date,
            'open': float(latest_data['Open']),
            'high': float(latest_data['High']),
            'low': float(latest_data['Low']),
            'close': float(latest_data['Close']),
            'volume': int(latest_data['Volume']),
            'adj_close': float(latest_data['Close']),
            'sma_5d': sma_5d,
            'monthly_trend': None
        }
        
    except Exception as e:
        print(f"❌ Error fetching today's data for {symbol}: {e}")
        return None

def insert_price_data(price_data):
    """Insert price data into the database"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Check if data for this date already exists
    cursor.execute("""
        SELECT id FROM prices 
        WHERE symbol = ? AND date = ?
    """, (price_data['symbol'], price_data['date']))
    
    existing = cursor.fetchone()
    
    if existing:
        print(f"⚠️ Data for {price_data['symbol']} on {price_data['date']} already exists")
        conn.close()
        return False
    
    # Insert new data
    cursor.execute("""
        INSERT INTO prices 
        (symbol, date, open, high, low, close, volume, adj_close, sma_5d, monthly_trend)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        price_data['symbol'],
        price_data['date'],
        price_data['open'],
        price_data['high'],
        price_data['low'],
        price_data['close'],
        price_data['volume'],
        price_data['adj_close'],
        price_data['sma_5d'],
        price_data['monthly_trend']
    ))
    
    conn.commit()
    conn.close()
    print(f"✅ Inserted data for {price_data['symbol']} on {price_data['date']}")
    return True

def main():
    print("🚀 Starting ETF price update for today from Yahoo Finance...")
    print("=" * 60)
    
    # Get list of ETFs
    etfs = get_etf_list()
    print(f"📊 Found {len(etfs)} ETFs to update")
    print(f"📅 Target date: {datetime.now().strftime('%Y-%m-%d')}")
    
    successful_updates = 0
    failed_updates = 0
    
    for symbol, name in etfs:
        print(f"\n📈 Fetching today's data for {symbol} - {name}")
        
        # Fetch data from Yahoo Finance
        price_data = get_today_price_data(symbol)
        
        if price_data:
            # Insert into database
            if insert_price_data(price_data):
                print(f"💰 Price: ${price_data['close']:.2f}, Volume: {price_data['volume']:,}")
                successful_updates += 1
        else:
            failed_updates += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print(f"🎯 Update complete!")
    print(f"✅ Successful updates: {successful_updates}")
    print(f"❌ Failed updates: {failed_updates}")
    print(f"📈 Total ETFs processed: {len(etfs)}")
    
    if successful_updates > 0:
        # Show the most recent updates
        print("\n📅 Most recent price updates:")
        conn = sqlite3.connect('etf_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, close, volume
            FROM prices 
            WHERE date = (SELECT MAX(date) FROM prices)
            ORDER BY symbol
            LIMIT 10
        """)
        
        recent_updates = cursor.fetchall()
        for symbol, date, close, volume in recent_updates:
            print(f"   {symbol}: ${close:.2f} on {date} (Vol: {volume:,})")
        
        conn.close()

if __name__ == "__main__":
    main()