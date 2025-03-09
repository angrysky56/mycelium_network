"""
Environment implementation for the mycelium network.

The environment provides the spatial context in which the mycelium network exists,
including resources, obstacles, and stimuli that affect network growth and behavior.
"""

import math
import random
from typing import Dict, List, Tuple, Set, Optional, Union

# Modified file: mycelium/environment.py

from mycelium.spatial.spatial_index import SpatialIndex

class Environment:
    """Represents the environment in which the mycelium network exists."""
    
    def __init__(self, dimensions: int = 2, size: float = 1.0):
        """
        Initialize the environment.
        
        Args:
            dimensions: Number of spatial dimensions
            size: Size of the environment in each dimension
        """
        self.dimensions = dimensions
        self.size = size
        
        # Create spatial indices for resources, stimuli, and obstacles
        bounds = ((0, 0), (size, size)) if dimensions == 2 else ((0, 0, 0), (size, size, size))
        self.resource_index = SpatialIndex(dimensions, bounds)
        self.stimuli_index = SpatialIndex(dimensions, bounds)
        self.obstacle_index = SpatialIndex(dimensions, bounds)
        
        # Legacy dictionaries (for backward compatibility)
        self.resources = {}  # Spatial resource distribution
        self.stimuli = {}    # Environmental stimuli
        self.obstacles = []  # Regions where growth is restricted
        
    def add_resource(self, position: Tuple[float, ...], amount: float):
        """Add a resource at a specific position."""
        # Add to spatial index
        resource_id = f"resource_{hash(position)}"
        self.resource_index.insert(resource_id, position, amount)
        
        # Update legacy dictionary
        self.resources[position] = amount
    
    def add_stimulus(self, position: Tuple[float, ...], stimulus_type: str, intensity: float):
        """Add an environmental stimulus."""
        # Add to spatial index
        stimulus_id = f"stimulus_{hash(position)}_{stimulus_type}"
        
        # If the position already has stimuli, update the data
        existing_data = {}
        for item_id, (pos, data) in self.stimuli_index.get_all_items().items():
            if pos == position:
                existing_data = data or {}
                break
        
        # Update with new stimulus
        existing_data[stimulus_type] = intensity
        self.stimuli_index.insert(stimulus_id, position, existing_data)
        
        # Update legacy dictionary
        if position not in self.stimuli:
            self.stimuli[position] = {}
        self.stimuli[position][stimulus_type] = intensity
    
    def add_obstacle(self, position: Tuple[float, ...], radius: float):
        """Add an obstacle that restricts growth."""
        # Add to spatial index
        obstacle_id = f"obstacle_{hash(position)}"
        self.obstacle_index.insert(obstacle_id, position, radius)
        
        # Update legacy list
        self.obstacles.append((position, radius))
    
    def get_resources_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], float]:
        """Get all resources within a certain radius of a position."""
        # Use spatial index for efficient query
        resources = self.resource_index.query_range(position, radius)
        
        # Convert to expected return format
        result = {}
        for _, (pos, amount) in resources.items():
            result[pos] = amount
            
        return result
    
    def get_stimuli_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], Dict[str, float]]:
        """Get all stimuli within a certain radius of a position."""
        # Use spatial index for efficient query
        stimuli = self.stimuli_index.query_range(position, radius)
        
        # Convert to expected return format
        result = {}
        for _, (pos, stim_dict) in stimuli.items():
            result[pos] = stim_dict
            
        return result
    
    def is_position_valid(self, position: Tuple[float, ...]) -> bool:
        """Check if a position is valid (within bounds and not in an obstacle)."""
        # Check if within bounds
        for coord in position:
            if coord < 0 or coord > self.size:
                return False
        
        # Check if in any obstacle using spatial index
        obstacles = self.obstacle_index.query_range(position, 0)
        for _, (obs_pos, obs_radius) in obstacles.items():
            if self.calculate_distance(position, obs_pos) <= obs_radius:
                return False
        
        return True
    
    # Other methods remain unchanged
    
    def __init__(self, dimensions: int = 2, size: float = 1.0):
        """
        Initialize the environment.
        
        Args:
            dimensions: Number of spatial dimensions
            size: Size of the environment in each dimension
        """
        self.dimensions = dimensions
        self.size = size
        self.resources = {}  # Spatial resource distribution
        self.stimuli = {}    # Environmental stimuli
        self.obstacles = []  # Regions where growth is restricted
        
    def add_resource(self, position: Tuple[float, ...], amount: float):
        """Add a resource at a specific position."""
        self.resources[position] = amount
    
    def add_stimulus(self, position: Tuple[float, ...], stimulus_type: str, intensity: float):
        """Add an environmental stimulus."""
        if position not in self.stimuli:
            self.stimuli[position] = {}
        self.stimuli[position][stimulus_type] = intensity
    
    def add_obstacle(self, position: Tuple[float, ...], radius: float):
        """Add an obstacle that restricts growth."""
        self.obstacles.append((position, radius))
    
    def get_resources_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], float]:
        """Get all resources within a certain radius of a position."""
        result = {}
        for res_pos, amount in self.resources.items():
            if self.calculate_distance(position, res_pos) <= radius:
                result[res_pos] = amount
        return result
    
    def get_stimuli_in_range(self, position: Tuple[float, ...], radius: float) -> Dict[Tuple[float, ...], Dict[str, float]]:
        """Get all stimuli within a certain radius of a position."""
        result = {}
        for stim_pos, stim_dict in self.stimuli.items():
            if self.calculate_distance(position, stim_pos) <= radius:
                result[stim_pos] = stim_dict
        return result
    
    def is_position_valid(self, position: Tuple[float, ...]) -> bool:
        """Check if a position is valid (within bounds and not in an obstacle)."""
        # Check if within bounds
        for coord in position:
            if coord < 0 or coord > self.size:
                return False
        
        # Check if in an obstacle
        for obs_pos, obs_radius in self.obstacles:
            if self.calculate_distance(position, obs_pos) <= obs_radius:
                return False
        
        return True
    
    def calculate_distance(self, pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
        """Calculate the Euclidean distance between two positions."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def get_random_position(self) -> Tuple[float, ...]:
        """Generate a random valid position in the environment."""
        while True:
            position = tuple(random.random() * self.size for _ in range(self.dimensions))
            if self.is_position_valid(position):
                return position
    
    def create_grid_resources(self, grid_size: int = 5, resource_value: float = 1.0):
        """Create a grid of resources for testing purposes."""
        step = self.size / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                position = (i * step + step/2, j * step + step/2)
                if self.is_position_valid(position):
                    self.add_resource(position, resource_value)

    def create_random_obstacles(self, num_obstacles: int = 3, max_radius: float = 0.1):
        """Create random obstacles in the environment."""
        for _ in range(num_obstacles):
            position = tuple(random.random() * self.size for _ in range(self.dimensions))
            radius = random.random() * max_radius
            self.add_obstacle(position, radius)


