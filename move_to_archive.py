#!/usr/bin/env python3
"""
Script to move unused files to the archive directory.
"""

import os
import shutil
from pathlib import Path

# Files to keep in the examples directory
files_to_keep = [
    "__pycache__",
    "archive",
    "datasets",
    "reorganized",
]

# Get the base directory
base_dir = Path(__file__).parent
examples_dir = base_dir / "examples"
archive_dir = examples_dir / "archive"

# Ensure archive directory exists
archive_dir.mkdir(exist_ok=True)

# Move files to archive
for item in os.listdir(examples_dir):
    item_path = examples_dir / item
    
    # Skip directories we want to keep
    if item in files_to_keep:
        continue
    
    # Move to archive
    if item_path.is_file():
        destination = archive_dir / item
        print(f"Moving {item} to archive...")
        shutil.move(str(item_path), str(destination))

print("Files moved to archive successfully!")
