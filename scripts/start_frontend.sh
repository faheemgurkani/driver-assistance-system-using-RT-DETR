#!/bin/bash
# Start Next.js Frontend

# Resolve repository root relative to this script
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/frontend" || {
  echo "❌ Failed to enter frontend directory"
  exit 1
}

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start development server
npm run dev
