#!/usr/bin/env python3

import http.server
import socketserver
import json
import sqlite3
from urllib.parse import urlparse, parse_qs
import os
from datetime import datetime

class PriceSearchHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/price_search.html':
            self.path = '/price_search.html'
            return super().do_GET()
        elif self.path == '/favicon.ico':
            self.send_response(204)  # No Content
            self.end_headers()
            return
        else:
            return super().do_GET()
    
    def do_POST(self):
        if self.path == '/search_price':
            self.handle_price_search()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_price_search(self):
        try:
            print("=== Price Search Request Received ===")
            # Read request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            print(f"Request data: {request_data}")
            
            date = request_data.get('date')
            symbol = request_data.get('symbol', '').upper().strip()
            
            print(f"Parsed - Date: {date}, Symbol: {symbol}")
            
            if not date or not symbol:
                print("Missing date or symbol")
                self.send_json_response({'error': 'Date and symbol are required'}, 400)
                return
            
            # Validate date format
            try:
                datetime.strptime(date, '%Y-%m-%d')
                print("Date format valid")
            except ValueError:
                print("Invalid date format")
                self.send_json_response({'error': 'Invalid date format. Use YYYY-MM-DD'}, 400)
                return
            
            # Search database
            print("Starting database search...")
            results = self.search_price_data(date, symbol)
            print(f"Search completed, found {len(results)} results")
            
            self.send_json_response({'results': results})
            
        except Exception as e:
            print(f"Error handling price search: {e}")
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': 'Internal server error'}, 500)
    
    def search_price_data(self, date, symbol):
        """Search for price data in the database"""
        try:
            # Connect to database
            db_path = 'etf_data.db'
            if not os.path.exists(db_path):
                print(f"Database {db_path} not found")
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            print(f"Searching for {symbol} on {date}")
            
            # Search for exact match first
            query = """
                SELECT symbol, date, open, high, low, close, volume, monthly_trend 
                FROM prices 
                WHERE symbol = ? AND date = ?
                ORDER BY date DESC
            """
            
            cursor.execute(query, (symbol, date))
            results = cursor.fetchall()
            print(f"Exact match results: {len(results)} rows")
            
            # If no exact match, try to find closest dates
            if not results:
                print("No exact match, searching for closest dates")
                query = """
                    SELECT symbol, date, open, high, low, close, volume, monthly_trend 
                    FROM prices 
                    WHERE symbol = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 5
                """
                cursor.execute(query, (symbol, date))
                results = cursor.fetchall()
                print(f"Closest dates results: {len(results)} rows")
            
            conn.close()
            
            # Convert to list of dictionaries
            price_data = []
            for row in results:
                price_data.append({
                    'symbol': row[0],
                    'date': row[1],
                    'open': row[2],
                    'high': row[3],
                    'low': row[4],
                    'close': row[5],
                    'volume': row[6],
                    'monthly_trend': row[7]
                })
            
            print(f"Returning {len(price_data)} price records")
            return price_data
            
        except Exception as e:
            print(f"Database error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))

def main():
    PORT = 8086
    
    try:
        with socketserver.TCPServer(("", PORT), PriceSearchHandler) as httpd:
            print(f"Price Search Server running at http://localhost:{PORT}")
            print(f"Database: etf_data.db")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()