"""
Tests for spatial indexing functionality.

This module tests the quadtree/octree spatial indexing implementations.
"""

import unittest
import math
import random
from mycelium.spatial.spatial_index import SpatialIndex, SpatialNode


class TestSpatialNode(unittest.TestCase):
    """Test cases for the SpatialNode class."""
    
    def test_initialization(self):
        """Test node initialization."""
        bounds = ((0.0, 0.0), (1.0, 1.0))
        node = SpatialNode(bounds, max_items=5, max_depth=3)
        
        self.assertEqual(node.bounds, bounds)
        self.assertEqual(node.max_items, 5)
        self.assertEqual(node.max_depth, 3)
        self.assertEqual(node.depth, 0)
        self.assertEqual(len(node.items), 0)
        self.assertTrue(node.is_leaf())
    
    def test_contains(self):
        """Test point containment check."""
        bounds = ((0.0, 0.0), (1.0, 1.0))
        node = SpatialNode(bounds)
        
        # Points inside bounds
        self.assertTrue(node.contains((0.5, 0.5)))
        self.assertTrue(node.contains((0.0, 0.0)))
        self.assertTrue(node.contains((1.0, 1.0)))
        
        # Points outside bounds
        self.assertFalse(node.contains((1.1, 0.5)))
        self.assertFalse(node.contains((0.5, -0.1)))
    
    def test_subdivision(self):
        """Test node subdivision."""
        bounds = ((0.0, 0.0), (1.0, 1.0))
        node = SpatialNode(bounds, max_items=2, max_depth=3)
        
        # Add items to trigger subdivision
        node.insert("item1", (0.25, 0.25), "data1")
        node.insert("item2", (0.75, 0.75), "data2")
        node.insert("item3", (0.1, 0.9), "data3")
        
        # Check that subdivision occurred
        self.assertFalse(node.is_leaf())
        self.assertEqual(len(node.children), 4)  # Quadtree in 2D
        
        # Check child bounds
        child_bounds = [child.bounds for child in node.children]
        expected_bounds = [
            ((0.0, 0.0), (0.5, 0.5)),  # Bottom-left
            ((0.5, 0.0), (1.0, 0.5)),  # Bottom-right
            ((0.0, 0.5), (0.5, 1.0)),  # Top-left
            ((0.5, 0.5), (1.0, 1.0)),  # Top-right
        ]
        
        for bounds in expected_bounds:
            self.assertIn(bounds, child_bounds)
    
    def test_query_range(self):
        """Test querying items within a range."""
        bounds = ((0.0, 0.0), (1.0, 1.0))
        node = SpatialNode(bounds, max_items=10, max_depth=3)
        
        # Add test items
        items = [
            ("item1", (0.2, 0.2), "data1"),
            ("item2", (0.4, 0.4), "data2"),
            ("item3", (0.6, 0.6), "data3"),
            ("item4", (0.8, 0.8), "data4"),
            ("item5", (0.1, 0.9), "data5"),
        ]
        
        for item_id, pos, data in items:
            node.insert(item_id, pos, data)
        
        # Query items within range
        center = (0.3, 0.3)
        radius = 0.25
        results = node.query_range(center, radius)
        
        # Check results
        expected_items = ["item1", "item2"]
        for item_id in expected_items:
            self.assertIn(item_id, results)
        
        self.assertEqual(len(results), len(expected_items))


class TestSpatialIndex(unittest.TestCase):
    """Test cases for the SpatialIndex class."""
    
    def test_initialization(self):
        """Test index initialization."""
        index = SpatialIndex(dimensions=2)
        
        self.assertEqual(index.dimensions, 2)
        self.assertEqual(len(index.all_items), 0)
    
    def test_insert_and_query(self):
        """Test inserting items and querying them."""
        index = SpatialIndex(dimensions=2)
        
        # Add test items
        items = [
            ("item1", (0.2, 0.2), "data1"),
            ("item2", (0.4, 0.4), "data2"),
            ("item3", (0.6, 0.6), "data3"),
            ("item4", (0.8, 0.8), "data4"),
            ("item5", (0.1, 0.9), "data5"),
        ]
        
        for item_id, pos, data in items:
            index.insert(item_id, pos, data)
        
        # Check that all items were added
        self.assertEqual(len(index.all_items), len(items))
        
        # Query items within range
        center = (0.3, 0.3)
        radius = 0.25
        results = index.query_range(center, radius)
        
        # Check results
        expected_items = ["item1", "item2"]
        for item_id in expected_items:
            self.assertIn(item_id, results)
        
        self.assertEqual(len(results), len(expected_items))
    
    def test_update(self):
        """Test updating item positions and data."""
        index = SpatialIndex(dimensions=2)
        
        # Add an item
        index.insert("item1", (0.2, 0.2), "data1")
        
        # Update position
        index.update("item1", (0.8, 0.8))
        
        # Verify position was updated
        self.assertEqual(index.all_items["item1"][0], (0.8, 0.8))
        self.assertEqual(index.all_items["item1"][1], "data1")  # Data preserved
        
        # Update both position and data
        index.update("item1", (0.5, 0.5), "updated_data")
        
        # Verify both were updated
        self.assertEqual(index.all_items["item1"][0], (0.5, 0.5))
        self.assertEqual(index.all_items["item1"][1], "updated_data")
        
        # Query to ensure spatial index was updated
        results = index.query_range((0.5, 0.5), 0.1)
        self.assertIn("item1", results)
    
    def test_remove(self):
        """Test removing items."""
        index = SpatialIndex(dimensions=2)
        
        # Add test items
        items = [
            ("item1", (0.2, 0.2), "data1"),
            ("item2", (0.4, 0.4), "data2"),
            ("item3", (0.6, 0.6), "data3"),
        ]
        
        for item_id, pos, data in items:
            index.insert(item_id, pos, data)
        
        # Remove one item
        success = index.remove("item2")
        
        # Check removal was successful
        self.assertTrue(success)
        self.assertEqual(len(index.all_items), 2)
        self.assertNotIn("item2", index.all_items)
        
        # Check query results after removal
        results = index.query_range((0.4, 0.4), 0.1)
        self.assertNotIn("item2", results)
    
    def test_performance(self):
        """Test performance with many items."""
        index = SpatialIndex(dimensions=2, max_items=10, max_depth=6)
        
        # Add many random items
        num_items = 1000
        for i in range(num_items):
            item_id = f"item{i}"
            position = (random.random(), random.random())
            data = f"data{i}"
            
            index.insert(item_id, position, data)
        
        # Verify all items were added
        self.assertEqual(len(index.all_items), num_items)
        
        # Perform range queries at random points
        for _ in range(10):
            center = (random.random(), random.random())
            radius = random.uniform(0.05, 0.2)
            
            results = index.query_range(center, radius)
            
            # Verify results by direct calculation
            expected_items = {}
            for item_id, (pos, data) in index.all_items.items():
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, pos)))
                if dist <= radius:
                    expected_items[item_id] = (pos, data)
            
            self.assertEqual(len(results), len(expected_items))
            for item_id in expected_items:
                self.assertIn(item_id, results)


class Test3DSpatialIndex(unittest.TestCase):
    """Test cases for 3D spatial indexing."""
    
    def test_octree(self):
        """Test 3D octree implementation."""
        index = SpatialIndex(dimensions=3, bounds=((0,0,0), (1,1,1)))
        
        # Add test items
        items = [
            ("item1", (0.2, 0.2, 0.2), "data1"),
            ("item2", (0.4, 0.4, 0.4), "data2"),
            ("item3", (0.6, 0.6, 0.6), "data3"),
            ("item4", (0.8, 0.8, 0.8), "data4"),
            ("item5", (0.1, 0.9, 0.5), "data5"),
        ]
        
        for item_id, pos, data in items:
            index.insert(item_id, pos, data)
        
        # Query items within range in 3D
        center = (0.3, 0.3, 0.3)
        radius = 0.25
        results = index.query_range(center, radius)
        
        # Check results
        expected_items = ["item1", "item2"]
        for item_id in expected_items:
            self.assertIn(item_id, results)
        
        self.assertEqual(len(results), len(expected_items))


if __name__ == '__main__':
    unittest.main()
