const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 3022;

// Force HTTPS in production
if (process.env.NODE_ENV === 'production') {
    app.use((req, res, next) => {
        if (req.headers['x-forwarded-proto'] !== 'https') {
            return res.redirect('https://' + req.headers.host + req.url);
        }
        next();
    });
}

// Serve static files
app.use(express.static('.'));

// Serve the main dashboard
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'all_etfs_grid.html'));
});

// API endpoint to get available symbols
app.get('/api/symbols', (req, res) => {
    try {
        // Read symbols from the dedicated symbols file
        const symbolsData = fs.readFileSync(path.join(__dirname, 'etf_symbols.json'), 'utf8');
        const symbols = JSON.parse(symbolsData);
        res.json(symbols);
    } catch (error) {
        console.error('Error reading symbols:', error);
        res.status(500).json({ error: 'Error reading symbols' });
    }
});

// API endpoint to get ETF data
app.get('/api/etf_data', (req, res) => {
    const symbol = req.query.symbol;
    if (!symbol) {
        return res.status(400).json({ error: 'Symbol parameter required' });
    }
    
    try {
        // Try to find the real ETF data file
        const dataFile = path.join(__dirname, `${symbol}_price_data.json`);
        
        if (fs.existsSync(dataFile)) {
            // Read the real ETF data from the JSON file
            const data = fs.readFileSync(dataFile, 'utf8');
            const etfData = JSON.parse(data);
            res.json(etfData);
        } else {
            // Fallback to dummy data if real data not found
            console.log(`Real data not found for ${symbol}, using dummy data`);
            
            const startDate = new Date('2023-01-01');
            const endDate = new Date('2024-01-01');
            const dates = [];
            const close = [];
            
            // Generate dates for the past year
            let currentDate = new Date(startDate);
            while (currentDate <= endDate) {
                if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) { // Skip weekends
                    dates.push(currentDate.toISOString().split('T')[0]);
                    // Generate random price between 50 and 150
                    const basePrice = 100 + (symbol.charCodeAt(0) + symbol.charCodeAt(1)) % 50;
                    const randomVariation = Math.random() * 20 - 10;
                    close.push(basePrice + randomVariation);
                }
                currentDate.setDate(currentDate.getDate() + 1);
            }
            
            // Calculate 5-day moving average
            const sma_5d = [];
            for (let i = 0; i < close.length; i++) {
                if (i < 4) {
                    sma_5d.push(null);
                } else {
                    const sum = close.slice(i - 4, i + 1).reduce((a, b) => a + b, 0);
                    sma_5d.push(sum / 5);
                }
            }
            
            res.json({
                dates: dates,
                close: close,
                sma_5d: sma_5d
            });
        }
        
    } catch (error) {
        console.error('Error reading ETF data:', error);
        res.status(500).json({ error: 'Error reading ETF data' });
    }
});

// API endpoints to serve JSON data files
app.get('/api/:filename', (req, res) => {
    const filename = req.params.filename;
    const filepath = path.join(__dirname, filename);
    
    if (fs.existsSync(filepath)) {
        try {
            const data = fs.readFileSync(filepath, 'utf8');
            const jsonData = JSON.parse(data);
            res.json(jsonData);
        } catch (error) {
            res.status(500).json({ error: 'Error reading file' });
        }
    } else {
        res.status(404).json({ error: 'File not found' });
    }
});

app.listen(port, () => {
    console.log(`Simple ETF Dashboard running at http://localhost:${port}`);
});