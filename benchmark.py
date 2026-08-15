#!/usr/bin/env python3
"""
Root Benchmark Entry Point for Active Noise Cancelling
Contributed by: shadcy (https://github.com/shadcy)
"""

import os
import sys

# Auto-detect and switch to local virtual environment if dependencies are missing
try:
    import numpy
except ImportError:
    venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python3"))
    if os.path.exists(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        print("Error: numpy is not installed. Please create .venv and run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# Delegate directly to benchmark CLI
from benchmark.run_benchmarks import main

if __name__ == "__main__":
    main()
