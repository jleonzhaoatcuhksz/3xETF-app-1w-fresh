#!/usr/bin/env python3
import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration
ETF_SYMBOLS = ['UPRO', 'TMF', 'SOXL', 'NUGT', 'LABD', 'LABU', 'FAS', 'FAZ', 'DRN', 'SOXS', 'TMV', 'TECL']
DB_PATH = 'etf_data.db'
LOG_FILE = 'price_update.log'

def get_last_update_date():
    """Get the most recent date from the prices table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM prices")
    last_date = cursor.fetchone()[0]
    conn.close()
    return last_date

def update_prices():
    """Update prices from last update date to current date"""
    last_date = get_last_update_date()
    start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Updating prices from {start_date} to {end_date}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for symbol in ETF_SYMBOLS:
        try:
            print(f"Fetching {symbol} data...")
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if data.empty:
                print(f"No new data for {symbol}")
                continue

            data.reset_index(inplace=True)

            if isinstance(data.columns, pd.MultiIndex):
                flattened_columns = []
                for column_tuple in data.columns:
                    column_parts = [str(part).strip() for part in column_tuple if part]
                    if len(column_parts) > 1 and column_parts[-1].upper() == symbol.upper():
                        column_parts = column_parts[:-1]
                    flattened_columns.append('_'.join(column_parts).lower().replace(' ', '_'))
                data.columns = flattened_columns
            else:
                data.columns = [str(column_name).strip().lower().replace(' ', '_') for column_name in data.columns]

            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for column_name in numeric_columns:
                if column_name in data.columns:
                    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

            for _, row in data.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime) else str(row['date'])
                open_price = float(row['open']) if not pd.isna(row['open']) else None
                high_price = float(row['high']) if not pd.isna(row['high']) else None
                low_price = float(row['low']) if not pd.isna(row['low']) else None
                close_price = float(row['close']) if not pd.isna(row['close']) else None
                volume_value = int(row['volume']) if not pd.isna(row['volume']) else None
                adj_close_value = row['adj_close'] if 'adj_close' in row else (row['adj close'] if 'adj close' in row else None)
                adj_close_price = float(adj_close_value) if adj_close_value is not None and not pd.isna(adj_close_value) else None

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO prices 
                    (symbol, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        date_str,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume_value,
                        adj_close_price
                    )
                )

            print(f"Updated {len(data)} records for {symbol}")

        except Exception as e:
            print(f"Error updating {symbol}: {e}")

        time.sleep(1)  # Be gentle with Yahoo Finance API
    
    conn.commit()
    conn.close()
    
    # Log the update
    with open(LOG_FILE, 'a') as f:
        f.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Price update complete")

if __name__ == "__main__":
    update_prices()