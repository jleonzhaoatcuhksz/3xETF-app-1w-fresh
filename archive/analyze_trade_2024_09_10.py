#!/usr/bin/env python3

import sqlite3
import pandas as pd

def analyze_trade():
    conn = sqlite3.connect('etf_data.db')
    
    # Get FAS and NUGT prices around 2024-09-10
    query = '''
    SELECT date, symbol, close, monthly_trend 
    FROM prices 
    WHERE symbol IN ('FAS', 'NUGT') 
    AND date BETWEEN '2024-09-05' AND '2024-09-15'
    ORDER BY date, symbol
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print('🔍 Analysis of 2024-09-10 Trade: FAS → NUGT')
    print('=' * 60)
    print('📊 Price and Monthly Trend Data:')
    print()
    
    for date in sorted(df['date'].unique()):
        print(f'📅 {date}:')
        day_data = df[df['date'] == date]
        for _, row in day_data.iterrows():
            trend_pct = row['monthly_trend'] * 100 if pd.notna(row['monthly_trend']) else None
            trend_str = f'{trend_pct:+.2f}%' if trend_pct is not None else 'N/A'
            print(f'  {row["symbol"]}: ${row["close"]:.2f} (Monthly Trend: {trend_str})')
        print()
    
    # Calculate the trade performance
    print('💰 Trade Analysis:')
    print('-' * 30)
    
    # Find FAS price on 2024-09-10 and NUGT price on same day
    fas_data = df[(df['symbol'] == 'FAS') & (df['date'] == '2024-09-10')]
    nugt_data = df[(df['symbol'] == 'NUGT') & (df['date'] == '2024-09-10')]
    
    if not fas_data.empty and not nugt_data.empty:
        fas_price = fas_data.iloc[0]['close']
        nugt_price = nugt_data.iloc[0]['close']
        fas_trend = fas_data.iloc[0]['monthly_trend'] * 100
        nugt_trend = nugt_data.iloc[0]['monthly_trend'] * 100
        
        print(f'📈 From ETF (FAS): ${fas_price:.2f}, Monthly Trend: {fas_trend:+.2f}%')
        print(f'📉 To ETF (NUGT): ${nugt_price:.2f}, Monthly Trend: {nugt_trend:+.2f}%')
        print()
        
        # Look at what happened after the trade
        print('📊 Performance after the trade:')
        
        # Get prices for next few days
        query_after = '''
        SELECT date, symbol, close 
        FROM prices 
        WHERE symbol IN ('FAS', 'NUGT') 
        AND date BETWEEN '2024-09-10' AND '2024-09-16'
        ORDER BY date, symbol
        '''
        
        df_after = pd.read_sql_query(query_after, conn)
        
        fas_prices = df_after[df_after['symbol'] == 'FAS'].set_index('date')['close']
        nugt_prices = df_after[df_after['symbol'] == 'NUGT'].set_index('date')['close']
        
        if '2024-09-10' in fas_prices.index and '2024-09-10' in nugt_prices.index:
            fas_start = fas_prices['2024-09-10']
            nugt_start = nugt_prices['2024-09-10']
            
            print(f'  2024-09-10 (Trade Day):')
            print(f'    FAS: ${fas_start:.2f}')
            print(f'    NUGT: ${nugt_start:.2f}')
            
            for date in ['2024-09-11', '2024-09-12', '2024-09-13']:
                if date in fas_prices.index and date in nugt_prices.index:
                    fas_price_day = fas_prices[date]
                    nugt_price_day = nugt_prices[date]
                    
                    fas_return = (fas_price_day / fas_start - 1) * 100
                    nugt_return = (nugt_price_day / nugt_start - 1) * 100
                    
                    print(f'  {date}:')
                    print(f'    FAS: ${fas_price_day:.2f} ({fas_return:+.2f}%)')
                    print(f'    NUGT: ${nugt_price_day:.2f} ({nugt_return:+.2f}%)')
                    print(f'    📊 NUGT vs FAS: {nugt_return - fas_return:+.2f}% advantage')
                    print()

if __name__ == "__main__":
    analyze_trade()