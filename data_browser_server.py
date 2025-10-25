#!/usr/bin/env python3

import http.server
import socketserver
import json
import sqlite3
import os
from datetime import datetime
import csv
import io

class DataBrowserHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/data_browser.html':
            self.path = '/data_browser.html'
            return super().do_GET()
        elif self.path == '/get_symbols':
            self.handle_get_symbols()
            return
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        else:
            return super().do_GET()
    
    def do_POST(self):
        if self.path == '/search_data':
            self.handle_search_data()
        elif self.path == '/export_data':
            self.handle_export_data()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_get_symbols(self):
        """Return list of all unique ETF symbols"""
        try:
            conn = sqlite3.connect('etf_data.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            self.send_json_response({'symbols': symbols})
            
        except Exception as e:
            print(f"Error getting symbols: {e}")
            self.send_json_response({'error': 'Internal server error'}, 500)
    
    def handle_search_data(self):
        """Handle data browser search requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            symbol = request_data.get('symbol')
            start_date = request_data.get('start_date')
            end_date = request_data.get('end_date')
            page = request_data.get('page', 1)
            per_page = request_data.get('per_page', 50)
            
            conn = sqlite3.connect('etf_data.db')
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT date, symbol, open, high, low, close, volume, monthly_trend FROM prices"
            count_query = "SELECT COUNT(*) FROM prices"
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                count_query += " WHERE " + " AND ".join(conditions)
            
            # Get total count
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Add pagination
            query += " ORDER BY date DESC, symbol LIMIT ? OFFSET ?"
            params.extend([per_page, (page - 1) * per_page])
            
            cursor.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    'date': row[0],
                    'symbol': row[1],
                    'open': row[2],
                    'high': row[3],
                    'low': row[4],
                    'close': row[5],
                    'volume': row[6],
                    'monthly_trend': row[7]
                })
            
            conn.close()
            
            self.send_json_response({
                'results': results,
                'total': total,
                'total_pages': (total + per_page - 1) // per_page
            })
            
        except Exception as e:
            print(f"Error in data browser search: {e}")
            self.send_json_response({'error': 'Internal server error'}, 500)
    
    def handle_export_data(self):
        """Handle data export requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            symbol = request_data.get('symbol')
            start_date = request_data.get('start_date')
            end_date = request_data.get('end_date')
            
            conn = sqlite3.connect('etf_data.db')
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT date, symbol, open, high, low, close, volume, monthly_trend FROM prices"
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY date DESC, symbol"
            
            cursor.execute(query, params)
            
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 
                'Volume', 'Monthly Trend'
            ])
            
            # Write data
            for row in cursor.fetchall():
                writer.writerow(row)
            
            conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/csv')
            self.send_header('Content-Disposition', 'attachment; filename="etf_data_export.csv"')
            self.end_headers()
            
            self.wfile.write(output.getvalue().encode('utf-8'))
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            self.send_json_response({'error': 'Internal server error'}, 500)
    
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
    PORT = 8087  # Different port from price search server
    
    try:
        with socketserver.TCPServer(("", PORT), DataBrowserHandler) as httpd:
            print(f"Data Browser Server running at http://localhost:{PORT}")
            print(f"Database: etf_data.db")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()