#!/usr/bin/env python3
"""
Script to update ETF prices with intraday data from Yahoo Finance
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

def get_intraday_price_data(symbol):
    """Fetch intraday price data for a given ETF symbol"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get intraday data for today
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Try to get intraday data with 1h interval
        hist = ticker.history(period='1d', interval='1h')
        
        if hist.empty:
            print(f"❌ No intraday data found for {symbol}")
            return None
        
        # Get the latest intraday data point
        latest_data = hist.iloc[-1]
        
        # For intraday data, we'll use the date from the timestamp
        data_timestamp = hist.index[-1]
        data_date = data_timestamp.strftime('%Y-%m-%d')
        
        # Check if we got today's data
        if data_date != today:
            print(f"⚠️ Got data for {data_date} instead of {today} for {symbol}")
            return None
        
        # Get the approximate time of the data
        data_time = data_timestamp.strftime('%H:%M')
        
        # For SMA, we'll use the 5-day daily data
        hist_5d = ticker.history(period='5d')
        sma_5d = float(hist_5d['Close'].mean()) if len(hist_5d) >= 5 else None
        
        return {
            'symbol': symbol,
            'date': data_date,
            'time': data_time,
            'open': float(latest_data['Open']),
            'high': float(latest_data['High']),
            'low': float(latest_data['Low']),
            'close': float(latest_data['Close']),
            'volume': int(latest_data['Volume']),
            'adj_close': float(latest_data['Close']),
            'sma_5d': sma_5d,
            'monthly_trend': None,
            'is_intraday': True
        }
        
    except Exception as e:
        print(f"❌ Error fetching intraday data for {symbol}: {e}")
        return None

def insert_intraday_data(price_data):
    """Insert intraday price data into the database with special handling"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # For intraday data, we'll create a special record
    # We'll use a special date format to indicate intraday data
    intraday_date = f"{price_data['date']}_INTRADAY"
    
    # Check if intraday data for this date already exists
    cursor.execute("""
        SELECT id FROM prices 
        WHERE symbol = ? AND date = ?
    """, (price_data['symbol'], intraday_date))
    
    existing = cursor.fetchone()
    
    if existing:
        # Update existing intraday record
        cursor.execute("""
            UPDATE prices 
            SET open = ?, high = ?, low = ?, close = ?, volume = ?, 
                adj_close = ?, sma_5d = ?, monthly_trend = ?
            WHERE symbol = ? AND date = ?
        """, (
            price_data['open'],
            price_data['high'],
            price_data['low'],
            price_data['close'],
            price_data['volume'],
            price_data['adj_close'],
            price_data['sma_5d'],
            price_data['monthly_trend'],
            price_data['symbol'],
            intraday_date
        ))
        print(f"🔄 Updated intraday data for {price_data['symbol']} at {price_data['time']}")
    else:
        # Insert new intraday record
        cursor.execute("""
            INSERT INTO prices 
            (symbol, date, open, high, low, close, volume, adj_close, sma_5d, monthly_trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            price_data['symbol'],
            intraday_date,
            price_data['open'],
            price_data['high'],
            price_data['low'],
            price_data['close'],
            price_data['volume'],
            price_data['adj_close'],
            price_data['sma_5d'],
            price_data['monthly_trend']
        ))
        print(f"✅ Inserted intraday data for {price_data['symbol']} at {price_data['time']}")
    
    conn.commit()
    conn.close()
    return True

def main():
    print("🚀 Starting ETF intraday price update from Yahoo Finance...")
    print("=" * 60)
    
    # Get list of ETFs
    etfs = get_etf_list()
    print(f"📊 Found {len(etfs)} ETFs to update")
    print(f"📅 Target date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    print("💡 Note: Intraday data will be marked with '_INTRADAY' suffix")
    
    successful_updates = 0
    failed_updates = 0
    
    for symbol, name in etfs:
        print(f"\n📈 Fetching intraday data for {symbol} - {name}")
        
        # Fetch data from Yahoo Finance
        price_data = get_intraday_price_data(symbol)
        
        if price_data:
            # Insert into database
            if insert_intraday_data(price_data):
                print(f"💰 Price: ${price_data['close']:.2f} at {price_data['time']}")
                successful_updates += 1
        else:
            failed_updates += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"🎯 Intraday update complete!")
    print(f"✅ Successful updates: {successful_updates}")
    print(f"❌ Failed updates: {failed_updates}")
    print(f"📈 Total ETFs processed: {len(etfs)}")
    
    if successful_updates > 0:
        # Show the intraday updates
        print("\n📅 Intraday price updates:")
        conn = sqlite3.connect('etf_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, close, volume
            FROM prices 
            WHERE date LIKE '%_INTRADAY'
            ORDER BY symbol
            LIMIT 10
        """)
        
        intraday_updates = cursor.fetchall()
        for symbol, date, close, volume in intraday_updates:
            print(f"   {symbol}: ${close:.2f} on {date} (Vol: {volume:,})")
        
        conn.close()

if __name__ == "__main__":
    main()