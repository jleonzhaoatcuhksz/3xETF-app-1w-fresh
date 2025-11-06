#!/usr/bin/env python3
"""
Script to check Yahoo Finance status and available data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_symbol(symbol):
    """Check what data is available for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        
        print(f"\n🔍 Checking {symbol}:")
        
        # Try different time periods
        periods = ['1d', '2d', '5d', '1mo']
        intervals = ['1m', '5m', '15m', '1h', '1d']
        
        for period in periods:
            for interval in intervals:
                try:
                    hist = ticker.history(period=period, interval=interval)
                    if not hist.empty:
                        latest_date = hist.index[-1].strftime('%Y-%m-%d %H:%M')
                        print(f"   {period} @ {interval}: {len(hist)} records, latest: {latest_date}")
                        break
                except:
                    continue
            
            # Also try without interval
            try:
                hist = ticker.history(period=period)
                if not hist.empty:
                    latest_date = hist.index[-1].strftime('%Y-%m-%d')
                    print(f"   {period}: {len(hist)} records, latest: {latest_date}")
            except Exception as e:
                print(f"   {period}: Error - {e}")
        
        # Check if there's any info
        try:
            info = ticker.info
            if info:
                print(f"   Info available: {info.get('longName', 'N/A')}")
        except:
            print("   No info available")
            
    except Exception as e:
        print(f"   Error checking {symbol}: {e}")

def main():
    print("🔧 Checking Yahoo Finance Data Availability")
    print("=" * 60)
    print(f"📅 Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Test with a few symbols
    test_symbols = ['SPY', 'QQQ', 'SPXU', 'UPRO', 'TQQQ']
    
    for symbol in test_symbols:
        check_symbol(symbol)
    
    print("\n" + "=" * 60)
    print("💡 Analysis:")
    print("- If latest dates are 2025-11-05: Data for today not available yet")
    print("- If latest dates are 2025-11-06: Today's data is available")
    print("- If no data: Yahoo Finance API issue")

if __name__ == "__main__":
    main()