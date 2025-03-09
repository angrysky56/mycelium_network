#!/usr/bin/env python3
"""
Simple script to combine all parts of the biomimetic_mycelium.py file.
"""

import os

# Paths to the part files
part_files = []
for i in range(1, 14):
    filename = f"biomimetic_mycelium_p{i}.py"
    if os.path.exists(filename):
        part_files.append(filename)

# Additional part file from the split
if os.path.exists("biomimetic_mycelium_p8_cont.py"):
    part_files.append("biomimetic_mycelium_p8_cont.py")

# Header content for the main file
header = """#!/usr/bin/env python3
\"\"\"
Biomimetic Mycelium Network Implementation

This script implements a more biologically accurate mycelium-inspired neural network
with enhanced adaptive properties, stress responses, resource efficiency,
and environmental adaptability based on real fungal behavior.

Key biological inspirations:
- Nutrient sensing and directed growth
- Anastomosis (hyphal fusion)
- Stress-induced adaptation
- Enzymatic degradation of obstacles
- Spatial memory formation

Author: Claude AI
Date: March 8, 2025
\"\"\"

import os
import sys
import random
import time
import math
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union, Callable
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium import MyceliumNode, Signal, Environment, AdvancedMyceliumNetwork
from mycelium.tasks.classifier import MyceliumClassifier
"""

# Combine all parts into the target file
with open("biomimetic_mycelium.py", "w") as outfile:
    # Write the header
    outfile.write(header + "\n\n")
    
    # Write each part content
    for part_file in part_files:
        if os.path.exists(part_file):
            with open(part_file, "r") as infile:
                outfile.write(infile.read() + "\n")

print("Successfully assembled biomimetic_mycelium.py")
