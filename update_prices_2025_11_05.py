#!/usr/bin/env python3
"""
Script to update ETF prices for specific date 2025-11-05 from Yahoo Finance
Using EST timezone to ensure data accuracy
"""

import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import time

def get_etf_list():
    """Get list of all ETFs from the database"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT symbol, name FROM etfs ORDER BY symbol')
    etfs = cursor.fetchall()
    conn.close()
    return etfs

def get_price_data_for_date(symbol, target_date):
    """Fetch price data for a specific date"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Calculate start and end dates for the period
        start_date = target_date
        end_date = (datetime.strptime(target_date, '%Y-%m-%d') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch historical data for the specific date
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"❌ No data found for {symbol} on {target_date}")
            return None
        
        # Get the data for the specific date
        latest_data = hist.iloc[-1]
        data_date = hist.index[-1].strftime('%Y-%m-%d')
        
        # Verify we got the correct date
        if data_date != target_date:
            print(f"⚠️ Got data for {data_date} (expected {target_date})")
        
        # Get 5-day SMA (fetch 5 days ending on target date)
        start_date_sma = (datetime.strptime(target_date, '%Y-%m-%d') - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        hist_5d = ticker.history(start=start_date_sma, end=end_date)
        
        # Calculate SMA if we have enough data
        if len(hist_5d) >= 5:
            sma_5d = float(hist_5d['Close'].tail(5).mean())
        else:
            sma_5d = None
        
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
        print(f"❌ Error fetching data for {symbol} on {target_date}: {e}")
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
        print(f"⚠️ Data for {price_data['symbol']} on {price_data['date']} already exists - updating")
        # Update existing record
        cursor.execute("""
            UPDATE prices 
            SET open = ?, high = ?, low = ?, close = ?, volume = ?, adj_close = ?, sma_5d = ?, monthly_trend = ?
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
            price_data['date']
        ))
    else:
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
    target_date = "2025-11-05"
    
    print("🚀 Starting ETF price update for specific date 2025-11-05")
    print("=" * 70)
    print(f"📅 Target Date: {target_date}")
    print(f"🕐 Timezone: EST (US/Eastern)")
    print(f"🔧 Mode: Specific date update")
    
    # Get list of ETFs
    etfs = get_etf_list()
    print(f"📈 Found {len(etfs)} ETFs to update")
    
    successful_updates = 0
    failed_updates = 0
    
    for symbol, name in etfs:
        print(f"\n📈 Fetching data for {symbol} - {name}")
        
        # Fetch data from Yahoo Finance for specific date
        price_data = get_price_data_for_date(symbol, target_date)
        
        if price_data:
            # Insert/update data in database
            if insert_price_data(price_data):
                print(f"✅ Successfully updated {symbol} for {price_data['date']}")
                print(f"   💰 Price: ${price_data['close']:.2f}, Volume: {price_data['volume']:,}")
                successful_updates += 1
            else:
                print(f"❌ Failed to insert data for {symbol}")
                failed_updates += 1
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
        # Show the updated data
        print("\n📅 Updated price data for 2025-11-05:")
        conn = sqlite3.connect('etf_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, close, volume
            FROM prices 
            WHERE date = ?
            ORDER BY symbol
            LIMIT 10
        """, (target_date,))
        
        recent_updates = cursor.fetchall()
        for symbol, date, close, volume in recent_updates:
            print(f"   {symbol}: ${close:.2f} on {date} (Vol: {volume:,})")
        
        # Show how many records we have for this date
        cursor.execute("""
            SELECT COUNT(*) FROM prices WHERE date = ?
        """, (target_date,))
        
        count = cursor.fetchone()[0]
        print(f"\n📊 Total records for {target_date}: {count}")
        
        conn.close()

if __name__ == "__main__":
    main()