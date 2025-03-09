"""
Spatial indexing implementation for the mycelium network.

This module provides efficient spatial indexing structures (quadtree/octree)
to accelerate spatial queries like finding nearby resources or nodes.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import math


class SpatialNode:
    """A node in the spatial tree (quadtree/octree)."""
    
    def __init__(self, bounds, max_items=10, max_depth=5):
        """
        Initialize a spatial node.
        
        Args:
            bounds: Boundaries of this node (min_coords, max_coords)
            max_items: Maximum items before subdivision
            max_depth: Maximum tree depth
        """
        self.bounds = bounds
        self.max_items = max_items
        self.max_depth = max_depth
        self.depth = 0
        self.items = {}  # {item_id: (position, data)}
        self.children = None  # Subdivided nodes if any
        
    def is_leaf(self):
        """Check if this is a leaf node."""
        return self.children is None
        
    def contains(self, position):
        """Check if this node contains the given position."""
        min_coords, max_coords = self.bounds
        return all(min_coord <= coord <= max_coord 
                  for min_coord, max_coord, coord in zip(min_coords, max_coords, position))
    
    def intersects(self, query_bounds):
        """Check if this node intersects with the query bounds."""
        min_coords, max_coords = self.bounds
        query_min, query_max = query_bounds
        
        return all(min_coord <= query_max[i] and max_coord >= query_min[i]
                  for i, (min_coord, max_coord) in enumerate(zip(min_coords, max_coords)))
    
    def subdivide(self):
        """Subdivide this node into children."""
        if not self.is_leaf() or self.depth >= self.max_depth:
            return
            
        min_coords, max_coords = self.bounds
        mid_coords = tuple((min_coord + max_coord) / 2 
                           for min_coord, max_coord in zip(min_coords, max_coords))
        
        dim = len(min_coords)
        self.children = []
        
        # Create children based on dimensionality
        if dim == 2:
            # Quadtree (2D)
            self.children = [
                # Bottom-left
                SpatialNode(
                    ((min_coords[0], min_coords[1]), (mid_coords[0], mid_coords[1])),
                    self.max_items, self.max_depth
                ),
                # Bottom-right
                SpatialNode(
                    ((mid_coords[0], min_coords[1]), (max_coords[0], mid_coords[1])),
                    self.max_items, self.max_depth
                ),
                # Top-left
                SpatialNode(
                    ((min_coords[0], mid_coords[1]), (mid_coords[0], max_coords[1])),
                    self.max_items, self.max_depth
                ),
                # Top-right
                SpatialNode(
                    ((mid_coords[0], mid_coords[1]), (max_coords[0], max_coords[1])),
                    self.max_items, self.max_depth
                )
            ]
        elif dim == 3:
            # Octree (3D) - 8 children
            for i in range(8):
                # Calculate bounds for each octant
                child_min = [
                    min_coords[0] if (i & 1) == 0 else mid_coords[0],
                    min_coords[1] if (i & 2) == 0 else mid_coords[1],
                    min_coords[2] if (i & 4) == 0 else mid_coords[2]
                ]
                child_max = [
                    mid_coords[0] if (i & 1) == 0 else max_coords[0],
                    mid_coords[1] if (i & 2) == 0 else max_coords[1],
                    mid_coords[2] if (i & 4) == 0 else max_coords[2]
                ]
                
                child = SpatialNode(
                    (tuple(child_min), tuple(child_max)),
                    self.max_items, self.max_depth
                )
                child.depth = self.depth + 1
                self.children.append(child)
        
        # Redistribute items to children
        items_copy = self.items.copy()
        self.items.clear()
        
        for item_id, (position, data) in items_copy.items():
            self._insert_to_child(item_id, position, data)
    
    def _insert_to_child(self, item_id, position, data):
        """Insert an item into the appropriate child."""
        for child in self.children:
            if child.contains(position):
                child.insert(item_id, position, data)
                return
        
        # If no child contains the position, keep in this node
        self.items[item_id] = (position, data)
    
    def insert(self, item_id, position, data):
        """Insert an item at the given position."""
        if not self.contains(position):
            return False
            
        if self.is_leaf():
            # Add to this node
            self.items[item_id] = (position, data)
            
            # Subdivide if needed
            if len(self.items) > self.max_items and self.depth < self.max_depth:
                self.subdivide()
                
            return True
        else:
            # Try to insert into children
            for child in self.children:
                if child.contains(position):
                    return child.insert(item_id, position, data)
            
            # If no child contains it, add to this node
            self.items[item_id] = (position, data)
            return True
    
    def remove(self, item_id):
        """Remove an item by ID."""
        if item_id in self.items:
            del self.items[item_id]
            return True
            
        if not self.is_leaf():
            for child in self.children:
                if child.remove(item_id):
                    return True
                    
        return False
    
    def update(self, item_id, position, data):
        """Update an item's position and data."""
        # Remove then re-insert
        self.remove(item_id)
        return self.insert(item_id, position, data)
    
    def query_range(self, center, radius):
        """Query items within a radius."""
        results = {}
        
        # Check if this node intersects with the query range
        query_min = tuple(c - radius for c in center)
        query_max = tuple(c + radius for c in center)
        query_bounds = (query_min, query_max)
        
        if not self.intersects(query_bounds):
            return results
            
        # Check items in this node
        for item_id, (position, data) in self.items.items():
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, position)))
            if distance <= radius:
                results[item_id] = (position, data)
        
        # Check children
        if not self.is_leaf():
            for child in self.children:
                child_results = child.query_range(center, radius)
                results.update(child_results)
                
        return results


class SpatialIndex:
    """
    Spatial index for efficient spatial queries.
    
    Implements a quadtree/octree for 2D/3D spatial indexing.
    """
    
    def __init__(self, dimensions, bounds=None, max_items=10, max_depth=5):
        """
        Initialize the spatial index.
        
        Args:
            dimensions: Number of spatial dimensions (2 or 3)
            bounds: Environment bounds ((min_x, min_y, ...), (max_x, max_y, ...))
            max_items: Maximum items in a node before subdivision
            max_depth: Maximum tree depth
        """
        self.dimensions = dimensions
        
        # Default bounds if not provided
        if bounds is None:
            min_bounds = tuple(0.0 for _ in range(dimensions))
            max_bounds = tuple(1.0 for _ in range(dimensions))
            bounds = (min_bounds, max_bounds)
            
        # Create root node
        self.root = SpatialNode(bounds, max_items, max_depth)
        
        # Track all items for fast iteration
        self.all_items = {}  # {item_id: (position, data)}
    
    def insert(self, item_id, position, data=None):
        """
        Insert an item at the given position.
        
        Args:
            item_id: Unique identifier for the item
            position: Spatial coordinates
            data: Associated data
            
        Returns:
            True if inserted successfully
        """
        # Ensure position has correct dimensionality
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")
            
        # Add to index and tracking dictionary
        success = self.root.insert(item_id, position, data)
        if success:
            self.all_items[item_id] = (position, data)
            
        return success
    
    def remove(self, item_id):
        """
        Remove an item by ID.
        
        Args:
            item_id: ID of the item to remove
            
        Returns:
            True if removed successfully
        """
        if item_id in self.all_items:
            del self.all_items[item_id]
            return self.root.remove(item_id)
            
        return False
    
    def update(self, item_id, position, data=None):
        """
        Update an item's position and data.
        
        Args:
            item_id: ID of the item to update
            position: New position
            data: New data (or None to keep existing)
            
        Returns:
            True if updated successfully
        """
        # If data not provided and item exists, keep existing data
        if data is None and item_id in self.all_items:
            _, data = self.all_items[item_id]
            
        # Update in index and tracking dictionary
        success = self.root.update(item_id, position, data)
        if success:
            self.all_items[item_id] = (position, data)
            
        return success
    
    def query_range(self, center, radius):
        """
        Query items within a radius of a center point.
        
        Args:
            center: Center position
            radius: Radius to search within
            
        Returns:
            Dictionary of {item_id: (position, data)}
        """
        return self.root.query_range(center, radius)
    
    def get_all_items(self):
        """Get all items in the index."""
        return self.all_items.copy()
