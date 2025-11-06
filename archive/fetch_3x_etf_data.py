import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

def calculate_technical_indicators(df):
    """Calculate technical indicators for the price data"""
    # Simple Moving Averages
    df['sma_5d'] = df['Close'].rolling(window=5).mean()
    df['sma_10d'] = df['Close'].rolling(window=10).mean() 
    df['sma_20d'] = df['Close'].rolling(window=20).mean()
    df['sma_50d'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12d'] = df['Close'].ewm(span=12).mean()
    df['ema_26d'] = df['Close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12d'] - df['ema_26d']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    
    # Price momentum
    df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Monthly trend (approximated as 21-day trend)
    df['monthly_trend'] = df['Close'] / df['Close'].shift(21) - 1
    
    # Volatility (20-day rolling standard deviation)
    df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    return df

def fetch_etf_data(symbol, start_date, end_date, max_retries=3):
    """Fetch data for a single ETF with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"  Fetching {symbol} (attempt {attempt + 1}/{max_retries})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
            
            if df.empty:
                print(f"  ⚠️ No data found for {symbol}")
                return None
            
            # Reset index to get date as a column
            df = df.reset_index()
            df['Symbol'] = symbol
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            print(f"  ✅ Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"  ❌ Error fetching {symbol} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"  ❌ Failed to fetch {symbol} after {max_retries} attempts")
                return None
    
    return None

def main():
    # Connect to database
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    # Get all ETF symbols from the database
    cursor.execute("SELECT symbol FROM etfs ORDER BY symbol")
    etf_symbols = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(etf_symbols)} ETFs to fetch data for:")
    print(", ".join(etf_symbols))
    print()
    
    # Set date range (10 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)  # Approximately 10 years
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    total_records = 0
    successful_etfs = 0
    failed_etfs = []
    
    # Fetch data for each ETF
    for i, symbol in enumerate(etf_symbols, 1):
        print(f"\n[{i}/{len(etf_symbols)}] Processing {symbol}...")
        
        # Fetch data
        df = fetch_etf_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            try:
                # Prepare data for insertion
                records_to_insert = []
                for _, row in df.iterrows():
                    record = (
                        symbol,
                        row['Date'].strftime('%Y-%m-%d'),
                        float(row['Open']) if pd.notna(row['Open']) else None,
                        float(row['High']) if pd.notna(row['High']) else None,
                        float(row['Low']) if pd.notna(row['Low']) else None,
                        float(row['Close']) if pd.notna(row['Close']) else None,
                        int(row['Volume']) if pd.notna(row['Volume']) else None,
                        float(row['sma_5d']) if pd.notna(row['sma_5d']) else None,
                        float(row['sma_10d']) if pd.notna(row['sma_10d']) else None,
                        float(row['sma_20d']) if pd.notna(row['sma_20d']) else None,
                        float(row['sma_50d']) if pd.notna(row['sma_50d']) else None,
                        float(row['ema_12d']) if pd.notna(row['ema_12d']) else None,
                        float(row['ema_26d']) if pd.notna(row['ema_26d']) else None,
                        float(row['rsi']) if pd.notna(row['rsi']) else None,
                        float(row['macd']) if pd.notna(row['macd']) else None,
                        float(row['macd_signal']) if pd.notna(row['macd_signal']) else None,
                        float(row['macd_histogram']) if pd.notna(row['macd_histogram']) else None,
                        float(row['bb_upper']) if pd.notna(row['bb_upper']) else None,
                        float(row['bb_middle']) if pd.notna(row['bb_middle']) else None,
                        float(row['bb_lower']) if pd.notna(row['bb_lower']) else None,
                        float(row['volume_sma']) if pd.notna(row['volume_sma']) else None,
                        float(row['momentum_5d']) if pd.notna(row['momentum_5d']) else None,
                        float(row['momentum_10d']) if pd.notna(row['momentum_10d']) else None,
                        float(row['momentum_20d']) if pd.notna(row['momentum_20d']) else None,
                        float(row['monthly_trend']) if pd.notna(row['monthly_trend']) else None,
                        float(row['volatility']) if pd.notna(row['volatility']) else None
                    )
                    records_to_insert.append(record)
                
                # Insert into database
                cursor.executemany("""
                    INSERT INTO prices (
                        symbol, date, open, high, low, close, volume,
                        sma_5d, sma_10d, sma_20d, sma_50d, ema_12d, ema_26d,
                        rsi, macd, macd_signal, macd_histogram,
                        bb_upper, bb_middle, bb_lower, volume_sma,
                        momentum_5d, momentum_10d, momentum_20d, monthly_trend, volatility
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_to_insert)
                
                conn.commit()
                total_records += len(records_to_insert)
                successful_etfs += 1
                print(f"  ✅ Inserted {len(records_to_insert)} records for {symbol}")
                
            except Exception as e:
                print(f"  ❌ Error inserting data for {symbol}: {str(e)}")
                failed_etfs.append(symbol)
        else:
            failed_etfs.append(symbol)
        
        # Small delay to avoid overwhelming Yahoo Finance
        time.sleep(0.5)
    
    # Final summary
    print("\n" + "=" * 80)
    print("DATA FETCH SUMMARY")
    print("=" * 80)
    print(f"Total ETFs processed: {len(etf_symbols)}")
    print(f"Successfully fetched: {successful_etfs}")
    print(f"Failed: {len(failed_etfs)}")
    print(f"Total records inserted: {total_records:,}")
    
    if failed_etfs:
        print(f"\nFailed ETFs: {', '.join(failed_etfs)}")
    
    # Verify final count
    cursor.execute("SELECT COUNT(*) FROM prices")
    final_count = cursor.fetchone()[0]
    print(f"\nFinal record count in prices table: {final_count:,}")
    
    # Show date range of data
    cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
    date_range = cursor.fetchone()
    if date_range[0]:
        print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    # Show records per ETF
    cursor.execute("""
        SELECT symbol, COUNT(*) as record_count 
        FROM prices 
        GROUP BY symbol 
        ORDER BY record_count DESC
    """)
    etf_counts = cursor.fetchall()
    
    print(f"\nRecords per ETF:")
    for symbol, count in etf_counts:
        print(f"  {symbol}: {count:,} records")
    
    conn.close()
    print("\n✅ Data fetch completed!")

if __name__ == "__main__":
    main()