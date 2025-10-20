#!/usr/bin/env python3

import sqlite3
import pandas as pd

def analyze_fas_decline():
    conn = sqlite3.connect('etf_data.db')
    
    # Get FAS prices from Sept 3-10, 2024
    query = '''
    SELECT date, symbol, close, monthly_trend, volume
    FROM prices 
    WHERE symbol = 'FAS'
    AND date BETWEEN '2024-09-03' AND '2024-09-10'
    ORDER BY date
    '''
    
    df = pd.read_sql_query(query, conn)
    
    print('🔍 Analysis: Why FAS Lost 6% Between Sept 3-10, 2024')
    print('=' * 60)
    print('📊 FAS Daily Performance:')
    print()
    
    if len(df) > 0:
        start_price = df.iloc[0]['close']
        
        for i, row in df.iterrows():
            date = row['date']
            price = row['close']
            trend = row['monthly_trend'] * 100 if pd.notna(row['monthly_trend']) else None
            volume = row['volume']
            
            # Calculate daily return from start
            return_from_start = (price / start_price - 1) * 100
            
            # Calculate daily return from previous day
            if i > 0:
                prev_price = df.iloc[i-1]['close']
                daily_return = (price / prev_price - 1) * 100
                daily_str = f'({daily_return:+.2f}%)'  
            else:
                daily_str = '(Start)'
            
            trend_str = f'{trend:+.2f}%' if trend is not None else 'N/A'
            
            print(f'📅 {date}: ${price:.2f} {daily_str} | Total: {return_from_start:+.2f}% | Trend: {trend_str} | Vol: {volume:,}')
    
    # Get broader market context - check other financial ETFs
    print('\n🏦 Financial Sector Context (Sept 3-10, 2024):')
    print('-' * 50)
    
    # Check other financial ETFs for context
    financial_etfs = ['FAS', 'FAZ', 'XLF']  # 3x bull, 3x bear, 1x financial sector
    
    query_context = '''
    SELECT date, symbol, close
    FROM prices 
    WHERE symbol IN ('FAS', 'FAZ', 'XLF')
    AND date IN ('2024-09-03', '2024-09-10')
    ORDER BY symbol, date
    '''
    
    df_context = pd.read_sql_query(query_context, conn)
    conn.close()
    
    for symbol in financial_etfs:
        symbol_data = df_context[df_context['symbol'] == symbol]
        if len(symbol_data) >= 2:
            start_price = symbol_data.iloc[0]['close']
            end_price = symbol_data.iloc[-1]['close']
            total_return = (end_price / start_price - 1) * 100
            
            etf_name = {
                'FAS': 'Financial Bull 3x',
                'FAZ': 'Financial Bear 3x', 
                'XLF': 'Financial Sector 1x'
            }.get(symbol, symbol)
            
            print(f'{symbol} ({etf_name}): ${start_price:.2f} → ${end_price:.2f} ({total_return:+.2f}%)')
    
    print('\n💡 Analysis:')
    print('-' * 20)
    print('The 6% loss occurred because:')
    print('1. 📉 FAS declined from $129.65 to $121.37 while being held (Sept 3-10)')
    print('2. 🎯 This was NOT a trading loss, but a holding period loss')
    print('3. 📊 The ML model correctly switched OUT of declining FAS on Sept 10')
    print('4. ⚠️  3x leveraged ETFs are extremely volatile and can lose value quickly')

if __name__ == "__main__":
    analyze_fas_decline()