"""
Run FastAPI server
"""

import uvicorn
from pathlib import Path
import sys

# Ensure backend directory is on sys.path so `api.*` imports resolve
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

