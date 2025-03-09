#!/usr/bin/env python3
"""
Simple test for spatial indexing without visualization.
"""

import os
import sys
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.spatial.spatial_index import SpatialIndex


def test_spatial_index():
    """Test spatial index functionality."""
    print("Testing Spatial Index")
    print("====================")
    
    # Create spatial index
    index = SpatialIndex(dimensions=2)
    
    # Add items
    for i in range(1000):
        pos = (random.random(), random.random())
        index.insert(f"item{i}", pos, f"data{i}")
    
    print(f"Added 1000 items to spatial index")
    
    # Test query
    center = (0.5, 0.5)
    
    # Try different radius values
    for radius in [0.1, 0.2, 0.3, 0.4, 0.5]:
        # Time the query
        start_time = time.time()
        results = index.query_range(center, radius)
        elapsed = time.time() - start_time
        
        # Print results
        print(f"Query with radius {radius}: {len(results)} items in {elapsed:.4f} seconds")
    
    # Test update
    item_id = "item5"
    old_pos = index.all_items[item_id][0]
    new_pos = (random.random(), random.random())
    
    print(f"Updating item position from {old_pos} to {new_pos}")
    index.update(item_id, new_pos)
    
    # Verify update
    updated_pos = index.all_items[item_id][0]
    print(f"New position after update: {updated_pos}")
    
    # Test remove
    print(f"Removing item {item_id}")
    index.remove(item_id)
    
    # Verify removal
    remaining = len(index.all_items)
    print(f"Items remaining after removal: {remaining}")
    
    # Final query to verify consistency
    results = index.query_range(center, 0.3)
    print(f"Final query with radius 0.3: {len(results)} items")


if __name__ == "__main__":
    test_spatial_index()
