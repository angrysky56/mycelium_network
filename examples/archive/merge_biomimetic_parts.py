#!/usr/bin/env python3
"""Script to merge all biomimetic mycelium parts into one file."""

import os

# Define the output path
output_path = "/home/ty/Repositories/ai_workspace/mycelium_network/examples/biomimetic_mycelium.py"

# Get all part files in the correct order
part_files = []
for i in range(1, 15):  # Parts 1 to 14
    part_path = f"/home/ty/Repositories/ai_workspace/mycelium_network/examples/biomimetic_mycelium_part{i}.py"
    if os.path.exists(part_path):
        part_files.append(part_path)

# Read each part and combine them
output_content = ""
for part_file in part_files:
    try:
        with open(part_file, 'r') as f:
            output_content += f.read() + "\n\n"
    except Exception as e:
        print(f"Error reading {part_file}: {e}")

# Write the combined content to the output file
try:
    with open(output_path, 'w') as f:
        f.write(output_content)
    print(f"Successfully created: {output_path}")
except Exception as e:
    print(f"Error writing output file: {e}")
