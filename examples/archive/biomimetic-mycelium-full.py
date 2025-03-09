#!/usr/bin/env python3
"""
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
"""

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


class EnzymaticAbility:
    """
    Represents an enzymatic capability that allows a node to break down obstacles
    or process specific resources in the environment.
    """
    
    def __init__(self, enzyme_type: str, strength: float = 0.0, energy_cost: float = 0.1):
        """
        Initialize enzymatic ability.
        
        Args:
            enzyme_type: Type of enzyme ('cellulase', 'lignin_peroxidase', etc.)
            strength: Effectiveness of the enzyme (0.0 to 1.0)
            energy_cost: Energy required to produce the enzyme
        """
        self.type = enzyme_type
        self.strength = strength
        self.energy_cost = energy_cost
        self.active = False
        self.production_rate = 0.01  # Base rate of enzyme production
        
    def activate(self, energy_available: float) -> float:
        """
        Activate enzyme production.
        
        Args:
            energy_available: Energy available for enzyme production
            
        Returns:
            Energy consumed
        """
        if energy_available >= self.energy_cost:
            self.active = True
            return self.energy_cost
        else:
            self.active = False
            return 0.0
            
    def deactivate(self):
        """Deactivate enzyme production."""
        self.active = False
        
    def apply_to_obstacle(self, obstacle_type: str) -> float:
        """
        Apply enzyme to break down an obstacle.
        
        Args:
            obstacle_type: Type of obstacle
            
        Returns:
            Effectiveness against this obstacle (0.0 to 1.0)
        """
        if not self.active:
            return 0.0
            
        # Match enzyme to obstacle type
        effectiveness = {
            'cellulose': {'cellulase': 0.8, 'beta_glucosidase': 0.5},
            'lignin': {'lignin_peroxidase': 0.7, 'manganese_peroxidase': 0.6},
            'protein': {'protease': 0.9},
            'lipid': {'lipase': 0.8}
        }
        
        return effectiveness.get(obstacle_type, {}).get(self.type, 0.1) * self.strength


class BiomimeticNode(MyceliumNode):
    """
    Enhanced mycelium node with more biologically accurate behaviors.
    """
    
    def __init__(self, node_id: int, position: Tuple[float, ...], node_type: str = 'regular'):
        """
        Initialize a biomimetic node.
        
        Args:
            node_id: Unique identifier for the node
            position: Spatial coordinates in the environment
            node_type: Type of node ('input', 'hidden', 'output', or 'regular')
        """
        super().__init__(node_id, position, node_type)
        
        # Enhanced biological properties
        self.hyphal_branches = []  # Sub-branches from this node
        self.anastomosis_targets = set()  # Nodes that this node has fused with
        self.spatial_memory = {}  # {position_hash: (resource_level, timestamp)}
        
        # Enzymatic capabilities
        self.enzymes = {
            'cellulase': EnzymaticAbility('cellulase', random.uniform(0.1, 0.4)),
            'lignin_peroxidase': EnzymaticAbility('lignin_peroxidase', random.uniform(0.0, 0.3)),
            'protease': EnzymaticAbility('protease', random.uniform(0.0, 0.2))
        }
        
        # Stress responses
        self.stress_level = 0.0
        self.stress_memory = []  # [(stress_type, level, timestamp),...]
        
        # Enhanced metabolic properties
        self.nutrient_storage = defaultdict(float)  # {nutrient_type: amount}
        self.metabolic_rate = random.uniform(0.05, 0.15)  # Base metabolic rate
        self.growth_potential = random.uniform(0.5, 1.5)  # Growth capacity
        
        # Circadian-like rhythm
        self.cycle_phase = random.uniform(0, 2 * math.pi)  # Random starting phase
        self.cycle_duration = random.randint(20, 40)  # Cycle length in iterations
        
    def sense_environment(self, environment: Environment, radius: float = 0.2) -> Dict:
        """
        Sense the surrounding environment for resources, obstacles, and signals.
        
        Args:
            environment: Environment object
            radius: Sensing radius
            
        Returns:
            Dictionary with sensed information
        """
        sensed_info = {
            'resources': [],
            'obstacles': [],
            'signals': [],
            'nodes': []
        }
        
        # Sense resources
        resources = environment.get_resources_in_radius(self.position, radius)
        sensed_info['resources'] = resources
        
        # Sense obstacles
        obstacles = environment.get_obstacles_in_radius(self.position, radius)
        sensed_info['obstacles'] = obstacles
        
        # Update spatial memory
        for resource in resources:
            pos_hash = hash(tuple(resource['position']))
            self.spatial_memory[pos_hash] = (resource['level'], time.time())
            
        # Record stress from obstacles
        for obstacle in obstacles:
            if obstacle['distance'] < 0.1:  # Close obstacle causes stress
                self.add_stress('obstacle', 0.2 * (1 - obstacle['distance'] / 0.1))
                
        return sensed_info
        
    def add_stress(self, stress_type: str, level: float) -> None:
        """
        Add a stress response.
        
        Args:
            stress_type: Type of stress ('drought', 'obstacle', 'toxin', etc.)
            level: Intensity of stress (0.0 to 1.0)
        """
        self.stress_level = min(1.0, self.stress_level + level)
        self.stress_memory.append((stress_type, level, time.time()))
        
        # Stress triggers adaptation
        if stress_type == 'drought':
            # Drought stress increases energy efficiency
            self.metabolic_rate = max(0.01, self.metabolic_rate * 0.9)
            
        elif stress_type == 'obstacle':
            # Obstacle stress activates enzymes
            for enzyme in self.enzymes.values():
                # Higher chance to activate relevant enzymes
                if random.random() < level * 0.5:
                    enzyme.activate(self.energy)
                    
        elif stress_type == 'toxin':
            # Toxin stress increases adaptability
            self.adaptability = min(2.0, self.adaptability * 1.1)
            # Emit warning signal
            warning = self.emit_signal('danger', level, {'type': 'toxin'})
            return warning
            
        # All stress gradually reduces as the node adapts
        self._adapt_to_stress()
        
    def _adapt_to_stress(self) -> None:
        """Gradually adapt to stress conditions."""
        # Decay old stress
        recent_stress = [s for s in self.stress_memory if time.time() - s[2] < 50]
        if recent_stress:
            # Calculate average recent stress
            avg_stress = sum(s[1] for s in recent_stress) / len(recent_stress)
            # Adaptive response proportional to stress level and adaptability
            adaptation_factor = 0.01 * self.adaptability * avg_stress
            
            # Adaptations
            self.sensitivity = min(2.0, self.sensitivity * (1 + adaptation_factor))
            self.longevity += int(10 * adaptation_factor)
        
        # Reduce current stress level
        self.stress_level = max(0.0, self.stress_level - 0.01)
        
    def attempt_anastomosis(self, other_node, success_probability: float = 0.3) -> bool:
        """
        Attempt to fuse with another node (hyphal anastomosis).
        
        Args:
            other_node: Target node for fusion
            success_probability: Base probability of successful fusion
            
        Returns:
            True if fusion successful, False otherwise
        """
        # Already fused
        if other_node.id in self.anastomosis_targets:
            return True
            
        # Factors affecting anastomosis success
        genetic_similarity = 0.8  # Same network = high similarity
        distance_factor = 1.0 - min(1.0, self.calculate_distance(other_node) / 0.2)
        energy_factor = min(1.0, (self.energy + other_node.energy) / 1.5)
        
        # Calculate success chance
        fusion_chance = success_probability * genetic_similarity * distance_factor * energy_factor
        
        # Attempt fusion
        if random.random() < fusion_chance:
            self.anastomosis_targets.add(other_node.id)
            other_node.anastomosis_targets.add(self.id)
            
            # Strengthen connection dramatically
            if other_node.id in self.connections:
                self.connections[other_node.id] *= 2.0
            else:
                self.connect_to(other_node, strength=0.5)
                
            # Energy cost for both nodes
            self.energy = max(0.1, self.energy - 0.15)
            other_node.energy = max(0.1, other_node.energy - 0.15)
            
            # Signal successful fusion
            fusion_signal = self.emit_signal(
                'anastomosis', 
                0.8, 
                {'target_id': other_node.id}
            )
            
            return True
        else:
            return False
            
    def calculate_distance(self, other_node) -> float:
        """
        Calculate Euclidean distance to another node.
        
        Args:
            other_node: Other node object
            
        Returns:
            Distance value
        """
        if len(self.position) != len(other_node.position):
            raise ValueError("Position dimensions don't match")
            
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other_node.position)))
        
    def process_signal(self, input_value: float) -> float:
        """
        Enhanced signal processing with circadian modulation.
        
        Args:
            input_value: Input signal value
            
        Returns:
            Output signal value
        """
        # Update circadian phase
        self.cycle_phase = (self.cycle_phase + 2 * math.pi / self.cycle_duration) % (2 * math.pi)
        
        # Circadian modulation of sensitivity (±20%)
        circadian_factor = 1.0 + 0.2 * math.sin(self.cycle_phase)
        
        # Stress reduces processing effectiveness
        stress_factor = 1.0 - (self.stress_level * 0.3)
        
        # Combined factors
        effective_sensitivity = self.sensitivity * circadian_factor * stress_factor
        
        # Apply sigmoid activation with effective sensitivity
        try:
            self.activation = 1 / (1 + math.exp(-input_value * effective_sensitivity))
        except OverflowError:
            self.activation = 0.0 if input_value < 0 else 1.0
        
        # Resource consumption with circadian and stress influences
        metabolic_demand = self.metabolic_rate * (1.0 + 0.2 * self.activation)
        
        # Circadian rhythm affects metabolism (reduced at "night")
        if math.sin(self.cycle_phase) < 0:
            metabolic_demand *= 0.7
            
        self.energy = max(0.1, self.energy - metabolic_demand)
        
        # Age the node
        self.age += 1
        
        return self.activation
        
    def allocate_resources(self, amount: float, resource_type: str = 'general') -> None:
        """
        Enhanced resource allocation with specific nutrient types.
        
        Args:
            amount: Amount of resources to receive
            resource_type: Type of resource ('carbon', 'nitrogen', 'phosphorus', 'general')
        """
        # Add to specific nutrient storage
        self.nutrient_storage[resource_type] += amount
        
        # Different nutrients have different effects
        if resource_type == 'carbon':
            # Carbon increases energy
            self.energy = min(1.5, self.energy + amount * 0.2)
            
        elif resource_type == 'nitrogen':
            # Nitrogen enhances enzyme production
            for enzyme in self.enzymes.values():
                enzyme.strength = min(1.0, enzyme.strength + amount * 0.1)
                
        elif resource_type == 'phosphorus':
            # Phosphorus enhances connection formation
            self.growth_potential += amount * 0.05
            
        else:  # General nutrients
            # General resources affect overall resource level
            self.resource_level = min(2.0, self.resource_level + amount)
            self.energy = min(1.0, self.energy + amount * 0.1)
        
        # Substantial resources extend longevity
        if amount > 0.3:
            self.longevity += 2
            
    def grow_hyphal_branch(self, environment: Environment, direction: Optional[Tuple[float, ...]] = None) -> Optional[int]:
        """
        Grow a new hyphal branch in a specific or random direction.
        
        Args:
            environment: Environment object
            direction: Optional growth direction vector
            
        Returns:
            ID of the new node if created, None otherwise
        """
        # Check if growth is possible
        if self.energy < 0.3 or self.resource_level < 0.5:
            return None
            
        # Determine growth direction
        if direction is None:
            # Random direction if none specified
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.05, 0.1) * self.growth_potential
            
            # In 2D space
            direction = (math.cos(angle), math.sin(angle))
        
        # Calculate new position
        new_position = tuple(p + d * 0.1 for p, d in zip(self.position, direction))
        
        # Ensure new position is within bounds
        new_position = tuple(max(0, min(1, p)) for p in new_position)
        
        # Check if position is valid in environment
        if not environment.is_position_valid(new_position):
            # Attempt to degrade obstacles
            obstacles = environment.get_obstacles_in_radius(new_position, 0.05)
            
            if obstacles:
                obstacle = obstacles[0]
                enzyme_type = self._select_enzyme_for_obstacle(obstacle['type'])
                
                if enzyme_type and self.enzymes[enzyme_type].active:
                    # Attempt to degrade obstacle
                    effectiveness = self.enzymes[enzyme_type].apply_to_obstacle(obstacle['type'])
                    
                    if random.random() < effectiveness:
                        # Successfully cleared obstacle
                        environment.remove_obstacle(obstacle['id'])
                    else:
                        # Failed to clear obstacle
                        return None
                else:
                    # No appropriate enzyme active
                    return None
        
        # Create new node with a new ID
        new_id = max(environment.get_highest_node_id() + 1, 1000 * self.id + len(self.hyphal_branches))
        
        # Create the node
        new_node = BiomimeticNode(new_id, new_position, 'regular')
        
        # Inherit some properties from parent
        new_node.sensitivity = self.sensitivity * random.uniform(0.9, 1.1)
        new_node.adaptability = self.adaptability * random.uniform(0.9, 1.1)
        
        # Initialize with some resources from parent
        resource_transfer = min(0.3, self.resource_level * 0.3)
        self.resource_level -= resource_transfer
        new_node.resource_level = resource_transfer
        
        energy_transfer = min(0.2, self.energy * 0.3)
        self.energy -= energy_transfer
        new_node.energy = energy_transfer
        
        # Connect to the parent node
        self.connect_to(new_node, strength=0.7)  # Strong initial connection
        
        # Add to hyphal branches
        self.hyphal_branches.append(new_id)
        
        return new_id
        
    def _select_enzyme_for_obstacle(self, obstacle_type: str) -> Optional[str]:
        """
        Select the most appropriate enzyme for an obstacle.
        
        Args:
            obstacle_type: Type of obstacle
            
        Returns:
            Enzyme type or None if no suitable enzyme
        """
        enzyme_mapping = {
            'cellulose': 'cellulase',
            'lignin': 'lignin_peroxidase',
            'protein': 'protease'
        }
        
        return enzyme_mapping.get(obstacle_type)
        
    def update_circadian_rhythm(self, external_cue: float = None) -> None:
        """
        Update the node's circadian rhythm, optionally synchronized to external cues.
        
        Args:
            external_cue: Optional external synchronization signal (0.0 to 1.0)
        """
        # Natural phase progression
        self.cycle_phase = (self.cycle_phase + 2 * math.pi / self.cycle_duration) % (2 * math.pi)
        
        # External synchronization if provided
        if external_cue is not None:
            # Convert external cue to target phase
            target_phase = external_cue * 2 * math.pi
            # Gradually adjust phase (entrainment)
            phase_difference = target_phase - self.cycle_phase
            # Normalize to -π to π
            if phase_difference > math.pi:
                phase_difference -= 2 * math.pi
            elif phase_difference < -math.pi:
                phase_difference += 2 * math.pi
                
            # Gradual adjustment
            adjustment = phase_difference * 0.1
            self.cycle_phase = (self.cycle_phase + adjustment) % (2 * math.pi)


class EnhancedEnvironment(Environment):
    """
    Enhanced environment with more biologically relevant features.
    """
    
    def __init__(self, dimensions: int = 2, size: float = 1.0):
        """
        Initialize the enhanced environment.
        
        Args:
            dimensions: Number of spatial dimensions
            size: Size of the environment (0.0 to size)
        """
        super().__init__(dimensions, size)
        
        # Environmental factors
        self.moisture_level = 0.7  # 0.0 (dry) to 1.0 (saturated)
        self.temperature = 0.5  # 0.0 (cold) to 1.0 (hot)
        self.ph_level = 0.5  # 0.0 (acidic) to 1.0 (alkaline)
        self.light_level = 0.6  # 0.0 (dark) to 1.0 (bright)
        
        # Resources and obstacles
        self.resources = []  # [{id, position, type, level},...]
        self.obstacles = []  # [{id, position, type, size},...]
        
        # Resource generation parameters
        self.resource_depletion_rate = 0.01
        self.resource_regeneration_rate = 0.005
        self.resource_clustering = 0.7  # Tendency of resources to cluster
        
        # Node registry
        self.nodes = {}  # {node_id: node}
        
        # Initialize environment
        self._initialize_environment()
        
    def _initialize_environment(self) -> None:
        """Initialize the environment with resources and obstacles."""
        # Create resource clusters
        self._create_resource_clusters(4, 5, 10)
        
        # Create obstacles
        self._create_obstacles(10)
        
    def _create_resource_clusters(self, num_clusters: int, resources_per_cluster: int, radius: float = 0.1) -> None:
        """
        Create clusters of resources.
        
        Args:
            num_clusters: Number of clusters to create
            resources_per_cluster: Resources per cluster
            radius: Cluster radius
        """
        resource_types = ['carbon', 'nitrogen', 'phosphorus', 'general']
        resource_id = 0
        
        for _ in range(num_clusters):
            # Cluster center
            center = self.get_random_position()
            main_type = random.choice(resource_types)
            
            # Create resources around center
            for _ in range(resources_per_cluster):
                # Random position within radius of center
                offset = [random.uniform(-radius, radius) for _ in range(self.dimensions)]
                position = tuple(min(self.size, max(0, c + o)) for c, o in zip(center, offset))
                
                # Resource properties
                if random.random() < 0.7:
                    resource_type = main_type  # Most resources in a cluster are the same type
                else:
                    resource_type = random.choice(resource_types)
                    
                level = random.uniform(0.3, 1.0)
                
                # Add resource
                self.resources.append({
                    'id': resource_id,
                    'position': position,
                    'type': resource_type,
                    'level': level,
                    'created_at': time.time()
                })
                resource_id += 1
    
    def _create_obstacles(self, num_obstacles: int) -> None:
        """
        Create obstacles in the environment.
        
        Args:
            num_obstacles: Number of obstacles to create
        """
        obstacle_types = ['cellulose', 'lignin', 'protein']
        obstacle_id = 0
        
        for _ in range(num_obstacles):
            position = self.get_random_position()
            obstacle_type = random.choice(obstacle_types)
            size = random.uniform(0.05, 0.15)
            
            self.obstacles.append({
                'id': obstacle_id,
                'position': position,
                'type': obstacle_type,
                'size': size
            })
            obstacle_id += 1
            
    def update(self) -> None:
        """Update the environment (resource regeneration, depletion, etc.)."""
        # Update environmental conditions
        self._update_environment_conditions()
        
        # Update resources
        self._update_resources()
        
    def _update_environment_conditions(self) -> None:
        """Update environmental conditions with small fluctuations."""
        # Small random fluctuations
        self.moisture_level = max(0.1, min(1.0, self.moisture_level + random.uniform(-0.03, 0.03)))
        self.temperature = max(0.1, min(1.0, self.temperature + random.uniform(-0.02, 0.02)))
        self.ph_level = max(0.1, min(1.0, self.ph_level + random.uniform(-0.01, 0.01)))
        
        # Light follows a day/night cycle
        day_cycle = (time.time() % 100) / 100  # 0.0 to 1.0
        self.light_level = 0.5 + 0.4 * math.sin(day_cycle * 2 * math.pi)
        
    def _update_resources(self) -> None:
        """Update resources (depletion, regeneration)."""
        # Deplete resources
        for resource in self.resources:
            resource['level'] = max(0.0, resource['level'] - self.resource_depletion_rate)
            
        # Remove depleted resources
        self.resources = [r for r in self.resources if r['level'] > 0.1]
        
        # Regenerate resources with small probability
        if random.random() < 0.05:
            self._create_resource_clusters(1, random.randint(2, 5))
            
    def get_resource_at_position(self, position: Tuple[float, ...]) -> Optional[Dict]:
        """
        Get resource at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Resource dictionary or None
        """
        for resource in self.resources:
            distance = self.calculate_distance(position, resource['position'])
            if distance < 0.05:  # Within resource radius
                return resource
        return None
        
    def get_resources_in_radius(self, center: Tuple[float, ...], radius: float) -> List[Dict]:
        """
        Get all resources within a radius.
        
        Args:
            center: Center position
            radius: Search radius
            
        Returns:
            List of resources with distance information
        """
        result = []
        for resource in self.resources:
            distance = self.calculate_distance(center, resource['position'])
            if distance <= radius:
                resource_copy = resource.copy()
                resource_copy['distance'] = distance
                result.append(resource_copy)
                
        # Sort by distance
        result.sort(key=lambda r: r['distance'])
        return result
        
    def get_obstacles_in_radius(self, center: Tuple[float, ...], radius: float) -> List[Dict]:
        """
        Get all obstacles within a radius.
        
        Args:
            center: Center position
            radius: Search radius
            
        Returns:
            List of obstacles with distance information
        """
        result = []
        for obstacle in self.obstacles:
            distance = self.calculate_distance(center, obstacle['position'])
            if distance <= radius + obstacle['size']:
                obstacle_copy = obstacle.copy()
                obstacle_copy['distance'] = distance
                result.append(obstacle_copy)
                
        # Sort by distance
        result.sort(key=lambda o: o['distance'])
        return result
        
    def remove_obstacle(self, obstacle_id: int) -> bool:
        """
        Remove an obstacle from the environment.
        
        Args:
            obstacle_id: ID of the obstacle to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, obstacle in enumerate(self.obstacles):
            if obstacle['id'] == obstacle_id:
                self.obstacles.pop(i)
                return True
        return False
        
    def consume_resource(self, resource_id: int, amount: float) -> float:
        """
        Consume some amount of a resource.
        
        Args:
            resource_id: ID of the resource
            amount: Amount to consume
            
        Returns:
            Amount actually consumed
        """
        for resource in self.resources:
            if resource['id'] == resource_id:
                consumed = min(resource['level'], amount)
                resource['level'] -= consumed
                return consumed
        return 0.0
        
    def is_position_valid(self, position: Tuple[float, ...]) -> bool:
        """
        Check if a position is valid (inside bounds and not inside an obstacle).
        
        Args:
            position: Position to check
            
        Returns:
            True if valid, False otherwise
        """
        # Check boundaries
        if not all(0 <= p <= self.size for p in position):
            return False
            
        # Check obstacles
        for obstacle in self.obstacles:
            distance = self.calculate_distance(position, obstacle['position'])
            if distance <= obstacle['size']:
                return False
                
        return True
        
    def get_environmental_conditions(self, position: Tuple[float, ...]) -> Dict[str, float]:
        """
        Get environmental conditions at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            Dictionary of conditions
        """
        # Base conditions
        conditions = {
            'moisture': self.moisture_level,
            'temperature': self.temperature,
            'ph': self.ph_level,
            'light': self.light_level
        }
        
        # Add spatial variations based on position
        x_factor = position[0] / self.size
        conditions['moisture'] = max(0.1, min(1.0, conditions['moisture'] * (1.0 - 0.3 * x_factor)))
        conditions['temperature'] = max(0.1, min(1.0, conditions['temperature'] * (1.0 + 0.2 * x_factor)))
        
        return conditions
        
    def register_node(self, node_id: int, node) -> None:
        """
        Register a node in the environment.
        
        Args:
            node_id: Node ID
            node: Node object
        """
        self.nodes[node_id] = node
        
    def get_highest_node_id(self) -> int:
        """
        Get the highest node ID in the environment.
        
        Returns:
            Highest node ID or 0 if no nodes
        """
        if not self.nodes:
            return 0
        return max(self.nodes.keys())


class BiomimeticNetwork(AdvancedMyceliumNetwork):
    """
    Enhanced mycelium network with more biologically accurate features and behaviors.
    """
    
    def __init__(
        self, 
        environment: EnhancedEnvironment = None,
        input_size: int = 3, 
        output_size: int = 1, 
        initial_nodes: int = 20
    ):
        """
        Initialize the biomimetic network.
        
        Args:
            environment: Enhanced environment in which the network exists
            input_size: Number of input nodes
            output_size: Number of output nodes
            initial_nodes: Initial number of regular nodes
        """
        # Create environment if not provided
        if environment is None:
            environment = EnhancedEnvironment()
            
        # Initialize parent
        super().__init__(environment, input_size, output_size, initial_nodes)

# Enhanced properties
        self.anastomosis_count = 0
        self.enzymatic_activity = defaultdict(float)  # {enzyme_type: activity_level}
        self.network_age = 0
        self.average_stress = 0.0
        self.resource_efficiency = 1.0
        self.adaptation_history = []
        
        # Replace regular nodes with biomimetic nodes
        self._convert_to_biomimetic_nodes()
        
        # Additional connections based on anastomosis
        self._initialize_hyphal_connections()
        
    def _convert_to_biomimetic_nodes(self) -> None:
        """Convert all nodes to biomimetic nodes."""
        for node_id, node in list(self.nodes.items()):
            # Convert only if not already biomimetic
            if not isinstance(node, BiomimeticNode):
                new_node = BiomimeticNode(node_id, node.position, node.type)
                
                # Transfer properties
                new_node.connections = node.connections.copy()
                new_node.activation = node.activation
                new_node.resource_level = node.resource_level
                new_node.energy = node.energy
                
                # Replace in network
                self.nodes[node_id] = new_node
                
                # Register with environment
                self.environment.register_node(node_id, new_node)
                
    def _initialize_hyphal_connections(self) -> None:
        """Initialize hyphal connections including anastomosis."""
        for node_id, node in self.nodes.items():
            # Skip input and output nodes
            if node_id in self.input_nodes or node_id in self.output_nodes:
                continue
                
            # Attempt anastomosis with nearby nodes
            for other_id, other_node in self.nodes.items():
                if (other_id not in self.input_nodes and 
                    other_id not in self.output_nodes and
                    other_id != node_id):
                    
                    # Check if close enough for anastomosis
                    if isinstance(node, BiomimeticNode) and isinstance(other_node, BiomimeticNode):
                        distance = node.calculate_distance(other_node)
                        
                        if distance < 0.15:  # Close enough for anastomosis
                            # Higher chance for nearby nodes
                            probability = 0.5 * (1 - distance / 0.15)
                            
                            if node.attempt_anastomosis(other_node, probability):
                                self.anastomosis_count += 1
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Enhanced forward pass with environmental sensing and adaptation.
        
        Args:
            inputs: Input values
            
        Returns:
            Output values
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Update environment
        self.environment.update()
        
        # Environmental sensing for all nodes
        self._sense_environment()
        
        # Enhanced forward pass
        outputs = super().forward(inputs)
        
        # Additional biomimetic behaviors
        self._perform_anastomosis()
        self._grow_hyphal_branches()
        self._update_enzyme_activities()
        self._process_circadian_rhythms()
        
        # Update network age
        self.network_age += 1
        
        return outputs
        
    def _sense_environment(self) -> None:
        """Have all nodes sense the environment."""
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                sensed_info = node.sense_environment(self.environment)
                
                # Process resources
                for resource in sensed_info['resources']:
                    if resource['distance'] < 0.05:  # Close enough to consume
                        consumed = self.environment.consume_resource(
                            resource['id'], 
                            min(0.2, node.energy * 0.5)
                        )
                        
                        if consumed > 0:
                            node.allocate_resources(consumed, resource['type'])
                
    def _perform_anastomosis(self) -> None:
        """Perform anastomosis between close nodes."""
        # Only attempt occasionally
        if random.random() > 0.1:
            return
            
        # Choose a random regular node
        if not self.regular_nodes:
            return
            
        node_id = random.choice(self.regular_nodes)
        node = self.nodes[node_id]
        
        if not isinstance(node, BiomimeticNode):
            return
            
        # Find nearby nodes
        for other_id, other_node in self.nodes.items():
            if (other_id != node_id and 
                other_id not in self.input_nodes and 
                other_id not in self.output_nodes and
                isinstance(other_node, BiomimeticNode)):
                
                distance = node.calculate_distance(other_node)
                
                if distance < 0.1 and random.random() < 0.3:
                    if node.attempt_anastomosis(other_node):
                        self.anastomosis_count += 1
                        
    def _grow_hyphal_branches(self) -> None:
        """Grow new hyphal branches from existing nodes."""
        # Only attempt occasionally
        if random.random() > 0.05:
            return
            
        # Only grow if sufficient network resources
        if self.total_resources < 8.0:
            return
            
        # Choose a node with high resources
        candidates = []
        for node_id in self.regular_nodes:
            node = self.nodes[node_id]
            if (isinstance(node, BiomimeticNode) and 
                node.resource_level > 1.2 and 
                node.energy > 0.6):
                candidates.append(node_id)
                
        if not candidates:
            return
            
        # Select a node to grow from
        node_id = random.choice(candidates)
        node = self.nodes[node_id]
        
        # Determine growth direction (towards resources if possible)
        direction = None
        resources = self.environment.get_resources_in_radius(node.position, 0.3)
        
        if resources:
            # Growth towards richest nearby resource
            resource = max(resources, key=lambda r: r['level'])
            # Calculate direction vector
            p1 = node.position
            p2 = resource['position']
            # Normalize direction vector
            magnitude = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            if magnitude > 0:
                direction = tuple((a - b) / magnitude for a, b in zip(p2, p1))
        
        # Grow new branch
        new_id = node.grow_hyphal_branch(self.environment, direction)
        
        if new_id is not None:
            # Add to network
            new_node = self.environment.nodes.get(new_id)
            if new_node:
                self.nodes[new_id] = new_node
                self.regular_nodes.append(new_id)
                self.total_resources -= 0.5
                
    def _update_enzyme_activities(self) -> None:
        """Update enzymatic activities across the network."""
        # Reset activity counters
        self.enzymatic_activity = defaultdict(float)
        
        # Sum up all enzyme activities
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                for enzyme_type, enzyme in node.enzymes.items():
                    if enzyme.active:
                        self.enzymatic_activity[enzyme_type] += enzyme.strength
        
    def _process_circadian_rhythms(self) -> None:
        """Process circadian rhythms across the network."""
        # Get light level as external synchronization cue
        light_level = self.environment.light_level
        
        # Update each node's circadian rhythm
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                node.update_circadian_rhythm(light_level)
                
    def train(self, 
              inputs: List[List[float]], 
              targets: List[List[float]], 
              epochs: int = 10, 
              learning_rate: float = 0.1) -> List[float]:
        """
        Train the network with enhanced biological learning mechanisms.
        
        Args:
            inputs: List of input vectors
            targets: List of target output vectors
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            
        Returns:
            List of error values for each epoch
        """
        epoch_errors = []
        
        for epoch in range(epochs):
            # Apply stress reduction for long-term training
            if epoch > epochs // 2:
                # Reduce stress in later epochs to stabilize learning
                for node_id, node in self.nodes.items():
                    if isinstance(node, BiomimeticNode) and node.stress_level > 0.3:
                        node.stress_level *= 0.9
            
            # Adaptive learning rate based on network age
            adaptive_rate = learning_rate * (1.0 / (1.0 + 0.05 * epoch))
            
            # Standard training
            epoch_error = super().train([inputs], [targets], 1, adaptive_rate)[0]
            epoch_errors.append(epoch_error)
            
            # Environmental adaptation after each epoch
            self.environment.update()
            self._sense_environment()
            
            # Track adaptation history
            self.adaptation_history.append({
                'epoch': epoch,
                'error': epoch_error,
                'nodes': len(self.nodes),
                'anastomosis': self.anastomosis_count,
                'resources': self.total_resources
            })
            
        return epoch_errors
        
    def get_network_statistics(self) -> Dict:
        """
        Get enhanced network statistics.
        
        Returns:
            Dictionary of network statistics
        """
        # Get basic statistics
        stats = super().get_network_statistics()
        
        # Add biomimetic statistics
        stats['anastomosis_count'] = self.anastomosis_count
        stats['network_age'] = self.network_age
        
        # Calculate average stress
        stress_sum = 0.0
        stress_count = 0
        
        for node_id, node in self.nodes.items():
            if isinstance(node, BiomimeticNode):
                stress_sum += node.stress_level
                stress_count += 1
                
        stats['average_stress'] = stress_sum / max(1, stress_count)
        
        # Enzymatic activities
        stats['enzymatic_activities'] = dict(self.enzymatic_activity)
        
        # Adaptation metrics
        if self.adaptation_history:
            stats['adaptation_progress'] = self.adaptation_history[-1]['error'] / max(0.001, self.adaptation_history[0]['error'])
            
        return stats
        
    def visualize_network(self, filename: str = None) -> Dict:
        """
        Enhanced visualization data for the network.
        
        Args:
            filename: If provided, save visualization data to this file
            
        Returns:
            Dictionary with visualization data
        """
        # Get basic visualization data
        vis_data = super().visualize_network(None)
        
        # Add biomimetic features
        for i, node_data in enumerate(vis_data['nodes']):
            node_id = node_data['id']
            node = self.nodes.get(node_id)
            
            if isinstance(node, BiomimeticNode):
                # Add biomimetic properties
                vis_data['nodes'][i].update({
                    'stress_level': node.stress_level,
                    'hyphal_branches': node.hyphal_branches,
                    'anastomosis_targets': list(node.anastomosis_targets),
                    'circadian_phase': node.cycle_phase
                })
                
        # Add environmental data
        vis_data['environment'] = {
            'moisture': self.environment.moisture_level,
            'temperature': self.environment.temperature,
            'ph': self.environment.ph_level,
            'light': self.environment.light_level,
            'resources': len(self.environment.resources),
            'obstacles': len(self.environment.obstacles)
        }
        
        # Save to file if specified
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(vis_data, f, indent=2)
        
        return vis_data
    
