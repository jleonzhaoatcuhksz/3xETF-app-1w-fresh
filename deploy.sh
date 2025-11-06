#!/bin/bash

# Set npm registry to avoid Tencent mirror issues
npm config set registry https://registry.npmjs.org/

# Install dependencies
npm install express sqlite3

# Start the application
node server_clean.js