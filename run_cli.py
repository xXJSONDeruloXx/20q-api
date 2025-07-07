#!/usr/bin/env python3
"""
Entry point for the 20Q-ANN CLI application.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from twentyq_ann.cli import cli

if __name__ == "__main__":
    cli()
