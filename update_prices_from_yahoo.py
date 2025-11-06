#!/usr/bin/env python3
"""
Script to update ETF prices from Yahoo Finance
Fetches the latest data for all ETFs in the etfs table and updates the prices table
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

def get_latest_price_data(symbol, days_back=7):
    """Fetch latest price data for a given ETF symbol"""
    try:
        # Get ETF data for the last few days to ensure we have today's data
        ticker = yf.Ticker(symbol)
        
        # Get historical data for the last week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            print(f"❌ No data found for {symbol}")
            return None
        
        # Get the most recent data (skip today if market not closed yet)
        latest_data = hist.iloc[-1]
        
        # Check if this is today's data or the last trading day
        data_date = hist.index[-1].strftime('%Y-%m-%d')
        
        # Get additional info
        info = ticker.info
        
        return {
            'symbol': symbol,
            'date': data_date,
            'open': float(latest_data['Open']),
            'high': float(latest_data['High']),
            'low': float(latest_data['Low']),
            'close': float(latest_data['Close']),
            'volume': int(latest_data['Volume']),
            'adj_close': float(latest_data['Close']),  # Yahoo doesn't always provide adj close
            'sma_5d': float(hist['Close'].tail(5).mean()) if len(hist) >= 5 else None,
            'monthly_trend': None,  # We'll calculate this later
            'name': info.get('longName', 'N/A')
        }
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
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
        print(f"⚠️ Data for {price_data['symbol']} on {price_data['date']} already exists, skipping")
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
    return True

def calculate_monthly_trends():
    """Calculate monthly trends for all ETFs"""
    conn = sqlite3.connect('etf_data.db')
    
    # Get all symbols
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM prices")
    symbols = [row[0] for row in cursor.fetchall()]
    
    for symbol in symbols:
        # Get last 30 days of data
        cursor.execute("""
            SELECT date, close 
            FROM prices 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 30
        """, (symbol,))
        
        data = cursor.fetchall()
        
        if len(data) < 2:
            continue
            
        # Calculate trend (current price vs 30 days ago)
        current_price = data[0][1]
        old_price = data[-1][1]
        
        if old_price > 0:
            trend = (current_price - old_price) / old_price
        else:
            trend = 0
        
        # Update the latest record with the trend
        latest_date = data[0][0]
        cursor.execute("""
            UPDATE prices 
            SET monthly_trend = ? 
            WHERE symbol = ? AND date = ?
        """, (trend, symbol, latest_date))
    
    conn.commit()
    conn.close()

def main():
    print("🚀 Starting ETF price update from Yahoo Finance...")
    print("=" * 60)
    
    # Get list of ETFs
    etfs = get_etf_list()
    print(f"📊 Found {len(etfs)} ETFs to update")
    
    successful_updates = 0
    failed_updates = 0
    
    for symbol, name in etfs:
        print(f"\n📈 Fetching data for {symbol} - {name}")
        
        # Fetch data from Yahoo Finance
        price_data = get_latest_price_data(symbol)
        
        if price_data:
            # Insert into database
            if insert_price_data(price_data):
                print(f"✅ Successfully updated {symbol} for {price_data['date']}")
                print(f"   Price: ${price_data['close']:.2f}, Volume: {price_data['volume']:,}")
                successful_updates += 1
            else:
                print(f"⚠️ Data already exists for {symbol}")
        else:
            failed_updates += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Calculate monthly trends
    print("\n📊 Calculating monthly trends...")
    calculate_monthly_trends()
    
    print("\n" + "=" * 60)
    print(f"🎯 Update complete!")
    print(f"✅ Successful updates: {successful_updates}")
    print(f"❌ Failed updates: {failed_updates}")
    print(f"📈 Total ETFs processed: {len(etfs)}")
    
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