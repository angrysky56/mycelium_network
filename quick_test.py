
#!/usr/bin/env python3
"""
Quick test to check if Python is working properly with our environment.
"""

import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python environment test")
print("======================")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print("Working directory:", os.getcwd())

try:
    from mycelium.enhanced.ml.genetic import NetworkGenome, GeneticOptimizer
    print("Successfully imported genetic module!")
except Exception as e:
    print(f"Error importing genetic module: {e}")

print("Test completed successfully!")
