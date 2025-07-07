#!/usr/bin/env python3
"""
Entry point for running the 20Q-ANN API server.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("twentyq_ann.api:app", host="127.0.0.1", port=8000, reload=True)
