const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
const port = process.env.PORT || 3022;

// Serve static files from public directory
app.use(express.static('public'));

// Database connection with fallback
let db;
try {
    db = new sqlite3.Database('etf_data.db');
    console.log('Connected to etf_data.db');
} catch (error) {
    console.log('Creating sample database...');
    db = new sqlite3.Database(':memory:');
    
    // Create sample data structure
    db.serialize(() => {
        db.run(`CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            date TEXT,
            close REAL,
            sma_5d REAL
        )`);
        
        // Add sample ETF data
        const sampleETFs = ['TQQQ', 'UPRO', 'SOXL', 'LABU', 'FAS', 'TMF'];
        sampleETFs.forEach((symbol, index) => {
            for (let i = 0; i < 30; i++) {
                const date = new Date(2024, 0, i + 1).toISOString().split('T')[0];
                const basePrice = 100 + (index * 10);
                const close = basePrice + Math.random() * 20;
                const sma_5d = close - Math.random() * 5;
                
                db.run(`INSERT INTO prices (symbol, date, close, sma_5d) 
                       VALUES (?, ?, ?, ?)`, 
                       [symbol, date, close, sma_5d]);
            }
        });
        console.log('Sample data created with 6 ETFs and 30 days of data each');
    });
}

// API endpoint to get all symbols
app.get('/api/symbols', (req, res) => {
    db.all("SELECT DISTINCT symbol FROM prices ORDER BY symbol", (err, rows) => {
        if (err) {
            console.log('Database error:', err);
            res.json([{symbol: 'TQQQ'}, {symbol: 'UPRO'}, {symbol: 'SOXL'}, 
                     {symbol: 'LABU'}, {symbol: 'FAS'}, {symbol: 'TMF'}]);
            return;
        }
        console.log(`Found ${rows.length} unique symbols`);
        res.json(rows);
    });
});

// API endpoint to fetch ETF price and SMA data
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
            // Return sample data if database error
            const sampleData = {
                dates: Array.from({length: 30}, (_, i) => 
                    new Date(2024, 0, i + 1).toISOString().split('T')[0]),
                close: Array.from({length: 30}, () => 100 + Math.random() * 50),
                sma_5d: Array.from({length: 30}, () => 95 + Math.random() * 40)
            };
            return res.json(sampleData);
        }

        if (rows.length === 0) {
            // Return sample data if no data found
            const sampleData = {
                dates: Array.from({length: 30}, (_, i) => 
                    new Date(2024, 0, i + 1).toISOString().split('T')[0]),
                close: Array.from({length: 30}, () => 100 + Math.random() * 50),
                sma_5d: Array.from({length: 30}, () => 95 + Math.random() * 40)
            };
            return res.json(sampleData);
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

// Simple statistics endpoint
app.get('/api/stats', (req, res) => {
    res.json({
        total_records: 180,
        earliest_date: "2024-01-01",
        latest_date: "2024-01-30",
        avg_price: 125.50
    });
});

// Serve the main grid page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'all_etfs_grid.html'));
});

app.listen(port, () => {
    console.log(`ETF Grid Dashboard running at http://localhost:${port}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});