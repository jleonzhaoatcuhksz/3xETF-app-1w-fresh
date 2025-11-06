const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 3022;

// Serve static files from public directory
app.use(express.static('public'));

// Database connection
const db = new sqlite3.Database('etf_data.db');

// API endpoint to get ETF data
app.get('/api/etfs', (req, res) => {
    db.all("SELECT id, symbol, date, close, sma_5d, monthly_trend FROM prices ORDER BY date DESC LIMIT 100", (err, rows) => {
        if (err) {
            console.log('Database error:', err);
            res.status(500).json({ error: err.message });
            return;
        }
        console.log(`Found ${rows.length} ETF price records`);
        res.json(rows);
    });
});

// API endpoint to get all symbols
app.get('/api/symbols', (req, res) => {
    db.all("SELECT DISTINCT symbol FROM prices ORDER BY symbol", (err, rows) => {
        if (err) {
            console.log('Database error:', err);
            res.status(500).json({ error: err.message });
            return;
        }
        console.log(`Found ${rows.length} unique symbols`);
        res.json(rows);
    });
});

// API endpoint to get ETF price and SMA data
app.get('/api/etf_data', (req, res) => {
    const { symbol } = req.query;
    if (!symbol) {
        return res.status(400).json({ error: 'Symbol parameter is required' });
    }

    const query = `
        SELECT date, close, sma_5d 
        FROM prices 
        WHERE symbol = ? 
        ORDER BY date ASC
    `;

    db.all(query, [symbol], (err, rows) => {
        if (err) {
            console.error('Database error:', err);
            return res.status(500).json({ error: err.message });
        }

        if (rows.length === 0) {
            return res.status(404).json({ error: 'No data found for this symbol' });
        }

        // Format data for Plotly
        const result = {
            dates: rows.map(row => row.date),
            close: rows.map(row => row.close),
            sma_5d: rows.map(row => row.sma_5d)
        };

        res.json(result);
    });
});

// Main route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
    console.log(`ETF Analysis Dashboard running at http://localhost:${port}`);
});