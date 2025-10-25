import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('etf_data.db')

print("SAMPLE DATA FROM PRICES TABLE")
print("=" * 80)

# Show recent data for a few different ETFs
sample_etfs = ['UPRO', 'TQQQ', 'SOXL', 'SPXU', 'SQQQ']

for etf in sample_etfs:
    print(f"\n📊 Recent data for {etf} (last 5 records):")
    print("-" * 60)
    
    query = """
        SELECT date, open, high, low, close, volume, sma_5d, monthly_trend
        FROM prices 
        WHERE symbol = ? 
        ORDER BY date DESC 
        LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn, params=(etf,))
    
    if not df.empty:
        # Format the data nicely
        df['open'] = df['open'].round(2)
        df['high'] = df['high'].round(2) 
        df['low'] = df['low'].round(2)
        df['close'] = df['close'].round(2)
        df['sma_5d'] = df['sma_5d'].round(2)
        df['monthly_trend'] = (df['monthly_trend'] * 100).round(2)  # Convert to percentage
        df['volume'] = df['volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
        
        print(f"{'Date':<12} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<12} {'SMA5d':<8} {'MonthlyTrend%':<12}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['date']:<12} {row['open']:<8} {row['high']:<8} {row['low']:<8} {row['close']:<8} {str(row['volume']):<12} {row['sma_5d']:<8} {row['monthly_trend']:<12}")
    else:
        print(f"No data found for {etf}")

# Show overall statistics
print(f"\n\n📈 OVERALL DATABASE STATISTICS")
print("=" * 80)

# Total records
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM prices")
total_records = cursor.fetchone()[0]

# Date range
cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
date_range = cursor.fetchone()

# ETF count
cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices")
etf_count = cursor.fetchone()[0]

# Records per ETF
cursor.execute("""
    SELECT symbol, COUNT(*) as count, MIN(date) as first_date, MAX(date) as last_date
    FROM prices 
    GROUP BY symbol 
    ORDER BY symbol
""")
etf_stats = cursor.fetchall()

print(f"Total records: {total_records:,}")
print(f"Date range: {date_range[0]} to {date_range[1]}")
print(f"Number of ETFs: {etf_count}")
print(f"Average records per ETF: {total_records // etf_count:,}")

print(f"\n📋 Records per ETF:")
print(f"{'Symbol':<6} {'Records':<8} {'First Date':<12} {'Last Date':<12}")
print("-" * 40)
for symbol, count, first_date, last_date in etf_stats:
    print(f"{symbol:<6} {count:<8,} {first_date:<12} {last_date:<12}")

# Show some price ranges to demonstrate volatility
print(f"\n💰 PRICE RANGES (demonstrating 3X ETF volatility)")
print("=" * 80)

cursor.execute("""
    SELECT symbol, 
           ROUND(MIN(close), 2) as min_price,
           ROUND(MAX(close), 2) as max_price,
           ROUND(AVG(close), 2) as avg_price,
           ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 1) as price_range_pct
    FROM prices 
    GROUP BY symbol 
    ORDER BY price_range_pct DESC
    LIMIT 10
""")
price_stats = cursor.fetchall()

print(f"{'Symbol':<6} {'Min Price':<10} {'Max Price':<10} {'Avg Price':<10} {'Range %':<10}")
print("-" * 50)
for symbol, min_price, max_price, avg_price, range_pct in price_stats:
    print(f"{symbol:<6} ${min_price:<9} ${max_price:<9} ${avg_price:<9} {range_pct}%")

conn.close()
print(f"\n✅ Sample data display completed!")