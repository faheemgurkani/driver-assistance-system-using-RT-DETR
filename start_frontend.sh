#!/bin/bash
# Start Next.js Frontend

cd "$(dirname "$0")/frontend"

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start development server
npm run dev

