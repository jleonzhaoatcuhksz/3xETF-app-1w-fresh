#!/usr/bin/env python3

import http.server
import socketserver
import webbrowser
import os
import json
from pathlib import Path

def start_dashboard_server():
    """Start a local HTTP server to serve the ML strategy dashboard"""
    
    # Change to the directory containing the files
    os.chdir(Path(__file__).parent)
    
    # Check if results file exists
    results_file = 'single_etf_ml_switching_results.json'
    if not os.path.exists(results_file):
        print(f"❌ Error: {results_file} not found!")
        print("Please run the ML strategy first to generate results.")
        return
    
    # Verify the results file is valid
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Found valid results file with {len(data.get('backtests', {}))} model results")
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return
    
    # Check if dashboard HTML exists
    dashboard_file = 'ml_strategy_dashboard.html'
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: {dashboard_file} not found!")
        return
    
    PORT = 8080
    
    # Custom handler to add CORS headers for local file access
    class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    # Try to find an available port
    for port in range(8080, 8090):
        try:
            with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
                PORT = port
                break
        except OSError:
            continue
    else:
        print("❌ Could not find an available port")
        return
    
    print("🚀 Starting ML ETF Strategy Dashboard Server...")
    print("=" * 60)
    print(f"📊 Server running at: http://localhost:{PORT}")
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"📈 Dashboard URL: http://localhost:{PORT}/ml_strategy_dashboard.html")
    print("=" * 60)
    print("💡 The dashboard will open automatically in your browser")
    print("💡 Press Ctrl+C to stop the server")
    print()
    
    # Open the dashboard in the default web browser
    dashboard_url = f"http://localhost:{PORT}/ml_strategy_dashboard.html"
    
    try:
        webbrowser.open(dashboard_url)
        print("✅ Dashboard opened in your default browser")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print(f"Please manually open: {dashboard_url}")
    
    # Start the server
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_dashboard_server()