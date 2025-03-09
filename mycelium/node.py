"""
Node and signal implementations for the mycelium network.

This module defines the basic building blocks of the mycelium network:
- Signal: Chemical signals that propagate through the network
- MyceliumNode: Individual nodes that form the network
"""

import math
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Union

from mycelium.environment import Environment


class Signal:
    """Represents a chemical signal in the mycelium network."""
    
    def __init__(self, signal_type: str, strength: float, source_id: int, metadata: Dict = None):
        """
        Initialize a new signal.
        
        Args:
            signal_type: Type of signal (e.g., 'nutrient', 'danger', 'reinforcement')
            strength: Initial signal strength
            source_id: ID of the node that generated the signal
            metadata: Additional signal information
        """
        self.type = signal_type
        self.strength = strength
        self.source_id = source_id
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.hops = 0  # Number of nodes this signal has traversed


class MyceliumNode:
    """
    Represents a node (hyphal tip or branch point) in the mycelium network.
    
    Each node has:
    - A spatial position
    - Connections to other nodes
    - Resource levels
    - Signaling capabilities
    - Growth potential
    """
    
    def __init__(self, node_id: int, position: Tuple[float, ...], node_type: str = 'regular'):
        """
        Initialize a mycelium node.
        
        Args:
            node_id: Unique identifier for the node
            position: Spatial coordinates in the environment
            node_type: Type of node ('input', 'hidden', 'output', or 'regular')
        """
        self.id = node_id
        self.position = position
        self.type = node_type
        
        # Connections to other nodes
        self.connections = {}  # {target_node_id: connection_strength}
        
        # Node state
        self.activation = 0.0
        self.resource_level = 1.0
        self.energy = 1.0
        self.age = 0
        self.longevity = random.randint(50, 200)  # Lifecycle in iterations
        self.sensitivity = random.uniform(0.8, 1.2)
        self.adaptability = random.uniform(0.5, 1.5)
        
        # Signal processing
        self.received_signals = []
        self.signal_memory = {}  # {signal_type: last_received_time}
        
        # Specialization (can develop over time)
        self.specializations = {}  # {specialization_type: level}
        
    def process_signal(self, input_value: float) -> float:
        """
        Process an input signal and update the node's activation.
        
        Args:
            input_value: Input signal value
            
        Returns:
            Output signal value
        """
        # Apply sigmoid activation with sensitivity factor
        try:
            self.activation = 1 / (1 + math.exp(-input_value * self.sensitivity))
        except OverflowError:
            self.activation = 0.0 if input_value < 0 else 1.0
        
        # Resource consumption proportional to activation
        energy_usage = self.activation * 0.05
        self.energy = max(0.1, self.energy - energy_usage)
        
        # Age the node
        self.age += 1
        
        return self.activation
    
    def receive_chemical_signal(self, signal: Signal) -> None:
        """
        Process an incoming chemical signal.
        
        Args:
            signal: The signal object
        """
        self.received_signals.append(signal)
        self.signal_memory[signal.type] = time.time()
        
        # Different responses based on signal type
        if signal.type == 'nutrient':
            # Nutrient signals increase resource level
            self.resource_level = min(2.0, self.resource_level + signal.strength * 0.2)
        elif signal.type == 'danger':
            # Danger signals increase sensitivity
            self.sensitivity = min(2.0, self.sensitivity + signal.strength * 0.1)
        elif signal.type == 'reinforcement':
            # Reinforcement signals enhance connections
            if 'connection_id' in signal.metadata:
                conn_id = signal.metadata['connection_id']
                if conn_id in self.connections:
                    self.connections[conn_id] *= (1 + signal.strength * 0.05)
    
    def emit_signal(self, signal_type: str, strength: float, metadata: Dict = None) -> Signal:
        """
        Emit a chemical signal.
        
        Args:
            signal_type: Type of signal to emit
            strength: Signal strength
            metadata: Additional signal information
            
        Returns:
            The created signal
        """
        signal = Signal(signal_type, strength, self.id, metadata)
        return signal
    
    def allocate_resources(self, amount: float) -> None:
        """
        Receive resources from the network or environment.
        
        Args:
            amount: Amount of resources to receive
        """
        self.resource_level = min(2.0, self.resource_level + amount)
        
        # Resources also restore some energy
        self.energy = min(1.0, self.energy + amount * 0.1)
        
        # High resource levels can extend longevity
        if self.resource_level > 1.5:
            self.longevity += 1
    
    def can_connect_to(self, other_node, environment: Environment, max_distance: float = 0.3) -> bool:
        """
        Determine if this node can connect to another node.
        
        Args:
            other_node: Potential target node
            environment: The environment context
            max_distance: Maximum allowed distance for connection
            
        Returns:
            True if connection is possible, False otherwise
        """
        if other_node.id == self.id:
            return False
            
        # Calculate distance in the environment
        distance = environment.calculate_distance(self.position, other_node.position)
        
        # Check if within range and has enough energy
        return (distance <= max_distance * self.resource_level and 
                self.energy > 0.3 and 
                self.is_path_clear(other_node, environment))
    
    def is_path_clear(self, other_node, environment: Environment) -> bool:
        """
        Check if the path to another node is clear of obstacles.
        
        Args:
            other_node: Target node
            environment: The environment context
            
        Returns:
            True if path is clear, False otherwise
        """
        # Simple implementation: check midpoint
        midpoint = tuple((a + b) / 2 for a, b in zip(self.position, other_node.position))
        return environment.is_position_valid(midpoint)
    
    def connect_to(self, other_node, strength: float = None) -> None:
        """
        Create or update a connection to another node.
        
        Args:
            other_node: Target node
            strength: Initial connection strength (if None, calculate based on distance)
        """
        if strength is None:
            # Start with a weak connection that can be strengthened through use
            strength = 0.1 + random.random() * 0.2
        
        # Store the connection
        self.connections[other_node.id] = strength
        
        # Creating connections uses energy
        self.energy = max(0.1, self.energy - 0.1)
    
    def develop_specialization(self, specialization_type: str, level_increase: float = 0.1) -> None:
        """
        Develop or enhance a specialization.
        
        Args:
            specialization_type: Type of specialization to develop
            level_increase: Amount to increase the specialization level
        """
        if specialization_type not in self.specializations:
            self.specializations[specialization_type] = 0.0
        
        self.specializations[specialization_type] = min(
            1.0, 
            self.specializations[specialization_type] + level_increase
        )
    
    def is_viable(self) -> bool:
        """
        Check if the node is still viable or should be removed.
        
        Returns:
            True if viable, False if should be removed
        """
        # Node dies if it reaches its longevity or has no energy
        return self.age < self.longevity and self.energy > 0.05
