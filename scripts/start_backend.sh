#!/bin/bash
# Start FastAPI Backend Server

# Resolve repository root relative to this script
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/backend" || {
  echo "❌ Failed to enter backend directory"
  exit 1
}

python api/run_server.py
