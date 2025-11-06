#!/usr/bin/env python3
"""
Script to update ETF prices from Yahoo Finance using EST timezone
"""

import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time

def get_est_time():
    """Get current time in EST"""
    est = pytz.timezone('US/Eastern')
    return datetime.now(est)

def get_etf_list():
    """Get list of all ETFs from the database"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT symbol, name FROM etfs ORDER BY symbol')
    etfs = cursor.fetchall()
    conn.close()
    return etfs

def is_market_open():
    """Check if the market is currently open (EST time)"""
    est_now = get_est_time()
    
    # Market hours: 9:30 AM - 4:00 PM EST, Monday-Friday
    market_open = est_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = est_now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if it's a weekday
    if est_now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if within market hours
    return market_open <= est_now <= market_close

def get_expected_trading_date():
    """Get the expected trading date based on EST time"""
    est_now = get_est_time()
    
    # If it's before market close, use today's date
    # If it's after market close, use the next trading day
    market_close = est_now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if est_now < market_close:
        # Market is open or will open today
        return est_now.strftime('%Y-%m-%d')
    else:
        # Market is closed, get next trading day
        next_day = est_now + timedelta(days=1)
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.strftime('%Y-%m-%d')

def get_latest_price_data_est(symbol):
    """Fetch latest price data for a given ETF symbol using EST logic"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current EST time
        est_now = get_est_time()
        
        # Determine what data to fetch based on market status
        if is_market_open():
            # Market is open - fetch intraday data
            print(f"   📊 Market is OPEN - fetching intraday data")
            hist = ticker.history(period='1d', interval='1h')
            
            if hist.empty:
                # Try daily data as fallback
                hist = ticker.history(period='2d')
        else:
            # Market is closed - fetch daily data
            print(f"   📊 Market is CLOSED - fetching daily data")
            hist = ticker.history(period='2d')  # Get last 2 days to ensure we have today
        
        if hist.empty:
            print(f"❌ No data found for {symbol}")
            return None
        
        # Get the most recent data
        latest_data = hist.iloc[-1]
        data_date = hist.index[-1].strftime('%Y-%m-%d')
        
        # Check if we got data for the expected date
        expected_date = get_expected_trading_date()
        if data_date != expected_date:
            print(f"⚠️ Got data for {data_date} (expected {expected_date})")
        
        # Get 5-day SMA
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
            'monthly_trend': None,
            'market_status': 'OPEN' if is_market_open() else 'CLOSED'
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
    return True

def main():
    print("🚀 Starting ETF price update from Yahoo Finance (EST Timezone)")
    print("=" * 70)
    
    # Get current EST time
    est_now = get_est_time()
    print(f"📅 EST Date: {est_now.strftime('%Y-%m-%d')}")
    print(f"⏰ EST Time: {est_now.strftime('%H:%M:%S')}")
    print(f"🏛️  Market Status: {'OPEN' if is_market_open() else 'CLOSED'}")
    print(f"📊 Expected Trading Date: {get_expected_trading_date()}")
    
    # Get list of ETFs
    etfs = get_etf_list()
    print(f"📈 Found {len(etfs)} ETFs to update")
    
    successful_updates = 0
    failed_updates = 0
    
    for symbol, name in etfs:
        print(f"\n📈 Fetching data for {symbol} - {name}")
        
        # Fetch data from Yahoo Finance
        price_data = get_latest_price_data_est(symbol)
        
        if price_data:
            # Insert into database
            if insert_price_data(price_data):
                print(f"✅ Successfully updated {symbol} for {price_data['date']}")
                print(f"   💰 Price: ${price_data['close']:.2f}, Volume: {price_data['volume']:,}")
                successful_updates += 1
        else:
            failed_updates += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
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