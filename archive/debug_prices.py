#!/usr/bin/env python3

import sqlite3

def debug_prices():
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()

    # Check TMF prices
    cursor.execute('SELECT date, close FROM prices WHERE symbol = ? AND date IN (?, ?) ORDER BY date', 
                   ('TMF', '2024-08-20', '2024-08-21'))
    tmf_prices = cursor.fetchall()
    print('TMF prices:')
    for date, price in tmf_prices:
        print(f'  {date}: ${price:.4f}')

    # Check SOXS prices  
    cursor.execute('SELECT date, close FROM prices WHERE symbol = ? AND date IN (?, ?) ORDER BY date', 
                   ('SOXS', '2024-08-20', '2024-08-21'))
    soxs_prices = cursor.fetchall()
    print('\nSOXS prices:')
    for date, price in soxs_prices:
        print(f'  {date}: ${price:.4f}')

    # Manual calculation
    print('\n=== MANUAL CALCULATION ===')
    print('Starting: 125.7747 UPRO shares at $79.51 = $10,000.00')
    
    print('\nTrade 1 (2024-08-20): UPRO -> TMF')
    upro_price = 79.5072250366211
    tmf_price_aug20 = 56.98616027832031
    upro_value = 125.7747 * upro_price
    tmf_shares = upro_value / tmf_price_aug20
    print(f'  Sold UPRO for: ${upro_value:.2f}')
    print(f'  Bought {tmf_shares:.4f} TMF shares at ${tmf_price_aug20:.4f}')
    print(f'  TMF portfolio value on 2024-08-20: ${tmf_shares * tmf_price_aug20:.2f}')
    
    # Get actual TMF price on Aug 21
    if len(tmf_prices) > 1:
        tmf_price_aug21 = tmf_prices[1][1]  # Second row, price column
        print(f'\nTrade 2 (2024-08-21): TMF -> SOXS')
        print(f'  TMF price changed to: ${tmf_price_aug21:.4f}')
        tmf_value_aug21 = tmf_shares * tmf_price_aug21
        print(f'  TMF portfolio value on 2024-08-21: ${tmf_value_aug21:.2f}')
        
        if len(soxs_prices) > 1:
            soxs_price_aug21 = soxs_prices[1][1]  # Second row, price column
            soxs_shares = tmf_value_aug21 / soxs_price_aug21
            print(f'  Bought {soxs_shares:.4f} SOXS shares at ${soxs_price_aug21:.4f}')
            print(f'  SOXS portfolio value on 2024-08-21: ${soxs_shares * soxs_price_aug21:.2f}')
            
            # This should NOT be $10,000 anymore if prices changed!
            print(f'\n🔍 EXPECTED: Portfolio value should change from $10,000 to ${tmf_value_aug21:.2f}')
    
    conn.close()

if __name__ == "__main__":
    debug_prices()