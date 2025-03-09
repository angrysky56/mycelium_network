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