import sqlite3
import json
import os
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect('etf_data.db')
cursor = conn.cursor()

# Get all ETF symbols from the database
cursor.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
symbols = [row[0] for row in cursor.fetchall()]

print(f"Found {len(symbols)} ETFs in the database:")
print(", ".join(symbols))

# Create the deploy-real-data directory if it doesn't exist
output_dir = 'deploy-real-data'
os.makedirs(output_dir, exist_ok=True)

# Extract data for each ETF
for symbol in symbols:
    print(f"\nExtracting data for {symbol}...")
    
    # Query to get all price data for this ETF
    query = """
        SELECT date, open, high, low, close, volume, sma_5d
        FROM prices 
        WHERE symbol = ? 
        ORDER BY date
    """
    
    cursor.execute(query, (symbol,))
    rows = cursor.fetchall()
    
    if not rows:
        print(f"  No data found for {symbol}")
        continue
    
    # Convert to the format expected by the frontend
    etf_data = {
        'dates': [],
        'close': [],
        'sma_5d': []
    }
    
    for date, open_price, high, low, close, volume, sma_5d in rows:
        etf_data['dates'].append(date)
        etf_data['close'].append(float(close) if close else 0)
        etf_data['sma_5d'].append(float(sma_5d) if sma_5d else 0)
    
    # Save as JSON file
    output_file = os.path.join(output_dir, f'{symbol}_price_data.json')
    with open(output_file, 'w') as f:
        json.dump(etf_data, f, indent=2)
    
    print(f"  Saved {len(rows)} records to {output_file}")

# Create the symbols list file
symbols_list = [{'symbol': symbol} for symbol in symbols]
symbols_file = os.path.join(output_dir, 'etf_symbols.json')
with open(symbols_file, 'w') as f:
    json.dump(symbols_list, f, indent=2)

print(f"\n✅ Created symbols file: {symbols_file}")

# Copy the HTML file and server.js to the new directory
import shutil

# Copy HTML file
html_source = 'deploy-simple/all_etfs_grid.html'
html_dest = os.path.join(output_dir, 'all_etfs_grid.html')
if os.path.exists(html_source):
    shutil.copy2(html_source, html_dest)
    print(f"  Copied HTML file to {html_dest}")

# Copy server.js
server_source = 'deploy-simple/server.js'
server_dest = os.path.join(output_dir, 'server.js')
if os.path.exists(server_source):
    shutil.copy2(server_source, server_dest)
    print(f"  Copied server.js to {server_dest}")

# Copy package.json
package_source = 'deploy-simple/package.json'
package_dest = os.path.join(output_dir, 'package.json')
if os.path.exists(package_source):
    shutil.copy2(package_source, package_dest)
    print(f"  Copied package.json to {package_dest}")

conn.close()
print(f"\n🎉 Real ETF data extraction completed!")
print(f"Output directory: {output_dir}")
print(f"Total ETFs processed: {len(symbols)}")