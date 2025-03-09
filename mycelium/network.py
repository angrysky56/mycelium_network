"""
Advanced Mycelium Network implementation.

This module implements the core functionality of the mycelium-inspired neural network,
including the network structure, signal propagation, and adaptive growth mechanics.
"""

import math
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Union, Callable

from mycelium.environment import Environment
from mycelium.node import MyceliumNode, Signal


class AdvancedMyceliumNetwork:
    """
    An advanced mycelium-inspired neural network with spatial awareness,
    chemical signaling, and adaptive growth.
    
    The network can be used for various tasks beyond traditional neural network
    applications, taking advantage of its adaptive and decentralized nature.
    """
    
    def __init__(
        self, 
        environment: Environment = None,
        input_size: int = 3, 
        output_size: int = 1, 
        initial_nodes: int = 20
    ):
        """
        Initialize the mycelium network.
        
        Args:
            environment: Environment in which the network exists
            input_size: Number of input nodes
            output_size: Number of output nodes
            initial_nodes: Initial number of regular nodes
        """
        self.environment = environment or Environment()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create node collections
        self.nodes = {}                 # All nodes by ID
        self.input_nodes = []           # Input node IDs
        self.output_nodes = []          # Output node IDs
        self.regular_nodes = []         # Regular node IDs
        
        # Track signals
        self.active_signals = []        # Currently active signals
        self.signal_history = []        # Record of past signals
        
        # Network state
        self.iteration = 0              # Current iteration/time step
        self.total_resources = 20.0     # Global resource pool
        self.growth_rate = 0.05         # Rate of new node formation
        self.adaptation_rate = 0.1      # Rate of network adaptation
        
        # Performance metrics
        self.error_history = []         # Training error history
        self.performance_metrics = {}   # Various performance measures
        
        # Initialize the network
        self._initialize_network(initial_nodes)
    
    def _initialize_network(self, initial_nodes: int) -> None:
        """
        Initialize the network with input, output, and regular nodes.
        
        Args:
            initial_nodes: Number of regular nodes to create initially
        """
        next_id = 0
        
        # Create input nodes (positioned in input region)
        for i in range(self.input_size):
            position = (0.1, 0.1 + (0.8 * i / max(1, self.input_size - 1)))
            node = MyceliumNode(next_id, position, node_type='input')
            self.nodes[next_id] = node
            self.input_nodes.append(next_id)
            next_id += 1
        
        # Create output nodes (positioned in output region)
        for i in range(self.output_size):
            position = (0.9, 0.1 + (0.8 * i / max(1, self.output_size - 1)))
            node = MyceliumNode(next_id, position, node_type='output')
            self.nodes[next_id] = node
            self.output_nodes.append(next_id)
            next_id += 1
        
        # Create initial regular nodes (positioned randomly)
        for i in range(initial_nodes):
            position = self.environment.get_random_position()
            node = MyceliumNode(next_id, position, node_type='regular')
            self.nodes[next_id] = node
            self.regular_nodes.append(next_id)
            next_id += 1
        
        # Create initial connections
        self._initialize_connections()
    
    def _initialize_connections(self) -> None:
        """Initialize the network with some basic connections."""
        # Connect each input node to some regular nodes
        for input_id in self.input_nodes:
            input_node = self.nodes[input_id]
            
            # Find closest regular nodes
            regular_distances = [
                (reg_id, self.environment.calculate_distance(
                    input_node.position, self.nodes[reg_id].position
                ))
                for reg_id in self.regular_nodes
            ]
            
            # Sort by distance and connect to the closest ones
            regular_distances.sort(key=lambda x: x[1])
            for reg_id, distance in regular_distances[:3]:  # Connect to 3 closest
                if input_node.can_connect_to(self.nodes[reg_id], self.environment):
                    input_node.connect_to(self.nodes[reg_id])
        
        # Connect regular nodes to each other based on proximity
        for i, node_id in enumerate(self.regular_nodes):
            node = self.nodes[node_id]
            
            # Find potential connections
            for other_id in self.regular_nodes[i+1:]:
                other_node = self.nodes[other_id]
                
                # Connect if possible and with some probability
                if (node.can_connect_to(other_node, self.environment) and 
                    random.random() < 0.3):
                    node.connect_to(other_node)
        
        # Connect some regular nodes to output nodes
        for output_id in self.output_nodes:
            output_node = self.nodes[output_id]
            
            # Find regular nodes that could connect to this output
            for reg_id in self.regular_nodes:
                reg_node = self.nodes[reg_id]
                
                if (reg_node.can_connect_to(output_node, self.environment) and 
                    random.random() < 0.2):
                    reg_node.connect_to(output_node)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input values
            
        Returns:
            Output values
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Clear previous activations and prepare signals
        for node in self.nodes.values():
            node.activation = 0.0
        
        self.active_signals = []
        
        # Feed inputs to input nodes and create initial signals
        for i, input_value in enumerate(inputs):
            input_id = self.input_nodes[i]
            self.nodes[input_id].activation = input_value
            
            # Create a nutrient signal based on input strength
            if input_value > 0.1:
                signal = self.nodes[input_id].emit_signal(
                    'nutrient', 
                    input_value, 
                    {'source': 'input'}
                )
                self.active_signals.append((input_id, signal))
        
        # Process activation through the network using signals
        visited_nodes = set(self.input_nodes)
        propagation_queue = deque([(input_id, 0) for input_id in self.input_nodes])
        
        # Breadth-first propagation of activation
        while propagation_queue:
            node_id, depth = propagation_queue.popleft()
            node = self.nodes[node_id]
            
            # Process outgoing connections
            for target_id, strength in node.connections.items():
                if target_id not in self.nodes:
                    continue  # Skip if target node doesn't exist
                
                target_node = self.nodes[target_id]
                
                # Calculate input signal for target node
                signal_value = node.activation * strength
                
                # Accumulate input at target
                target_node.activation += signal_value
                
                # Add to queue if not visited
                if target_id not in visited_nodes:
                    visited_nodes.add(target_id)
                    propagation_queue.append((target_id, depth + 1))
                    
                # Create a signal if significant activation
                if signal_value > 0.2:
                    signal = node.emit_signal(
                        'activation', 
                        signal_value, 
                        {'connection_id': target_id}
                    )
                    self.active_signals.append((node_id, signal))
        
        # Process signals through the network
        self._process_signals()
        
        # Process hidden nodes - apply activation function
        for node_id in self.regular_nodes:
            self.nodes[node_id].process_signal(self.nodes[node_id].activation)
        
        # Process output nodes
        outputs = []
        for output_id in self.output_nodes:
            output_value = self.nodes[output_id].process_signal(
                self.nodes[output_id].activation
            )
            outputs.append(output_value)
        
        # Update network state
        self.iteration += 1
        self._update_network_state()
        
        return outputs
    
    def _process_signals(self) -> None:
        """Process all active chemical signals in the network."""
        if not self.active_signals:
            return
        
        # Keep track of which nodes have received signals this cycle
        signal_receivers = defaultdict(list)
        
        # Process each signal
        next_signals = []
        for source_id, signal in self.active_signals:
            source_node = self.nodes[source_id]
            
            # Find nodes that can receive this signal
            for target_id, strength in source_node.connections.items():
                if target_id not in self.nodes:
                    continue
                
                target_node = self.nodes[target_id]
                
                # Signal weakens with distance
                distance = self.environment.calculate_distance(
                    source_node.position, target_node.position
                )
                attenuated_strength = signal.strength * math.exp(-2 * distance)
                
                if attenuated_strength > 0.05:  # Only propagate significant signals
                    # Create a weakened version of the signal
                    new_signal = Signal(
                        signal.type,
                        attenuated_strength,
                        source_id,
                        signal.metadata
                    )
                    
                    # Target receives the signal
                    target_node.receive_chemical_signal(new_signal)
                    signal_receivers[target_id].append(new_signal)
                    
                    # Signals can propagate further if still strong enough
                    if attenuated_strength > 0.2:
                        next_signals.append((target_id, new_signal))
        
        # Update active signals for next iteration
        self.active_signals = next_signals
        
        # Record significant signals for analysis
        for signal_source, signal in self.active_signals:
            if signal.strength > 0.3:
                self.signal_history.append((
                    self.iteration, 
                    signal_source, 
                    signal.type, 
                    signal.strength
                ))
    
    def _update_network_state(self) -> None:
        """Update the network state after a forward pass."""
        # Distribute global resources
        self._allocate_resources()
        
        # Consider growing new nodes or connections
        if random.random() < self.growth_rate:
            self._grow_network()
        
        # Remove dead or weak nodes
        self._prune_network()
        
        # Adapt connections based on recent activity
        self._adapt_connections()
    
    def _allocate_resources(self) -> None:
        """Allocate resources throughout the network."""
        # Calculate activity level for each node
        activity_levels = {}
        total_activity = 0
        
        for node_id, node in self.nodes.items():
            # Activity based on activation and signal processing
            activity = node.activation + 0.1 * len(node.received_signals)
            activity_levels[node_id] = activity
            total_activity += activity
        
        # Distribute resources proportionally to activity
        if total_activity > 0:
            resources_per_node = {}
            for node_id, activity in activity_levels.items():
                proportion = activity / total_activity
                resources_per_node[node_id] = self.total_resources * proportion * 0.1
            
            # Allocate to each node
            for node_id, amount in resources_per_node.items():
                self.nodes[node_id].allocate_resources(amount)
    
    def _grow_network(self) -> None:
        """Consider growing new nodes or connections in the network."""
        # Check the resource level - need sufficient resources to grow
        if self.total_resources < 5.0:
            return
        
        # Find nodes with high resource levels that could spawn new growth
        growth_candidates = []
        for node_id, node in self.nodes.items():
            if node.resource_level > 1.5 and node.energy > 0.7:
                growth_candidates.append(node_id)
        
        if not growth_candidates:
            return
        
        # Randomly select a node to grow from
        source_id = random.choice(growth_candidates)
        source_node = self.nodes[source_id]
        
        # Create a new node in a random direction
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.05, 0.2)
        
        # Calculate new position (2D for simplicity)
        new_x = source_node.position[0] + distance * math.cos(angle)
        new_y = source_node.position[1] + distance * math.sin(angle)
        new_position = (max(0, min(1, new_x)), max(0, min(1, new_y)))
        
        # Check if position is valid
        if not self.environment.is_position_valid(new_position):
            return
        
        # Create new node
        new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        new_node = MyceliumNode(new_id, new_position, 'regular')
        
        # Add to network
        self.nodes[new_id] = new_node
        self.regular_nodes.append(new_id)
        
        # Connect to source node
        source_node.connect_to(new_node, strength=0.5)  # Stronger initial connection
        
        # Connect to nearby nodes
        for node_id, node in self.nodes.items():
            if node_id != new_id and node_id != source_id:
                if new_node.can_connect_to(node, self.environment, max_distance=0.15):
                    if random.random() < 0.3:  # Only connect to some nearby nodes
                        new_node.connect_to(node)
        
        # Use resources for growth
        self.total_resources -= 1.0
        source_node.resource_level -= 0.5
        source_node.energy -= 0.3
    
    def _prune_network(self) -> None:
        """Remove dead or weak nodes and connections."""
        # Check each node for viability
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            # Don't remove input or output nodes
            if node_id in self.input_nodes or node_id in self.output_nodes:
                continue
            
            if not node.is_viable():
                nodes_to_remove.append(node_id)
        
        # Remove dead nodes
        for node_id in nodes_to_remove:
            if node_id in self.nodes:
                # Remove from collections
                self.nodes.pop(node_id)
                if node_id in self.regular_nodes:
                    self.regular_nodes.remove(node_id)
                
                # Remove connections to this node
                for other_node in self.nodes.values():
                    if node_id in other_node.connections:
                        del other_node.connections[node_id]
        
        # Prune weak connections
        for node in self.nodes.values():
            weak_connections = []
            for target_id, strength in node.connections.items():
                if strength < 0.05:
                    weak_connections.append(target_id)
            
            # Remove weak connections
            for target_id in weak_connections:
                if target_id in node.connections:
                    del node.connections[target_id]
    
    def _adapt_connections(self) -> None:
        """Adapt connection strengths based on recent activity."""
        # Strengthen connections that were active
        for node_id, node in self.nodes.items():
            for target_id, strength in list(node.connections.items()):
                if target_id not in self.nodes:
                    continue
                
                target_node = self.nodes[target_id]
                
                # Check if this connection was used effectively
                if (node.activation > 0.3 and target_node.activation > 0.3):
                    # Strengthen the connection
                    node.connections[target_id] *= 1.02
                else:
                    # Weaken unused connections
                    node.connections[target_id] *= 0.99
    
    def train(self, 
              inputs: List[List[float]], 
              targets: List[List[float]], 
              epochs: int = 10, 
              learning_rate: float = 0.1) -> List[float]:
        """
        Train the network using a combination of backpropagation-inspired
        learning and adaptive growth/pruning.
        
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
            epoch_error = 0.0
            
            # Create training pairs and shuffle
            training_data = list(zip(inputs, targets))
            random.shuffle(training_data)
            
            for input_vector, target_vector in training_data:
                # Forward pass
                outputs = self.forward(input_vector)
                
                # Calculate error
                errors = [t - o for t, o in zip(target_vector, outputs)]
                sample_error = sum(e**2 for e in errors) / len(errors)
                epoch_error += sample_error
                
                # Send reinforcement signals based on error
                for i, error in enumerate(errors):
                    output_id = self.output_nodes[i]
                    output_node = self.nodes[output_id]
                    
                    # Error signal
                    error_signal = output_node.emit_signal(
                        'error', 
                        abs(error), 
                        {'target': target_vector[i], 'output': outputs[i]}
                    )
                    self.active_signals.append((output_id, error_signal))
                    
                    # Process the error signal
                    self._process_signals()
                    
                    # Adjust connections based on error direction
                    for source_id, strength in self._get_incoming_connections(output_id).items():
                        source_node = self.nodes[source_id]
                        current_weight = strength
                        
                        # Weight update based on error and source activation
                        weight_change = learning_rate * error * source_node.activation
                        
                        # Apply the update
                        new_weight = max(0.01, min(2.0, current_weight + weight_change))
                        source_node.connections[output_id] = new_weight
                        
                        # Send reinforcement signal if error reduced
                        if sample_error < 0.1:
                            reinforcement = source_node.emit_signal(
                                'reinforcement',
                                0.5,
                                {'connection_id': output_id}
                            )
                            self.active_signals.append((source_id, reinforcement))
            
            # Calculate average error for this epoch
            avg_error = epoch_error / len(inputs)
            epoch_errors.append(avg_error)
            
            # Adapt learning rate based on progress
            if epoch > 0 and avg_error > epoch_errors[-2]:
                learning_rate *= 0.8  # Reduce learning rate if error increases
            elif epoch > 0 and avg_error < 0.8 * epoch_errors[-2]:
                learning_rate *= 1.05  # Slightly increase if error decreases significantly
            
            # More aggressive growth/pruning in early epochs
            if epoch < epochs // 3:
                self.growth_rate = 0.1
            else:
                self.growth_rate = 0.05
        
        return epoch_errors
    
    def _get_incoming_connections(self, node_id: int) -> Dict[int, float]:
        """
        Get all connections that point to a specific node.
        
        Args:
            node_id: Target node ID
            
        Returns:
            Dictionary of {source_id: connection_strength}
        """
        incoming = {}
        for source_id, node in self.nodes.items():
            if node_id in node.connections:
                incoming[source_id] = node.connections[node_id]
        return incoming
    
    def visualize_network(self, filename: str = None) -> Dict:
        """
        Generate data for visualizing the network.
        
        Args:
            filename: If provided, save visualization data to this file
            
        Returns:
            Dictionary with visualization data
        """
        # Generate visualization data
        vis_data = {
            'nodes': [],
            'connections': [],
            'metrics': {
                'total_nodes': len(self.nodes),
                'input_nodes': len(self.input_nodes),
                'output_nodes': len(self.output_nodes),
                'regular_nodes': len(self.regular_nodes),
                'total_connections': sum(len(node.connections) for node in self.nodes.values()),
                'iteration': self.iteration,
            }
        }
        
        # Add node data
        for node_id, node in self.nodes.items():
            node_type = 'input' if node_id in self.input_nodes else 'output' if node_id in self.output_nodes else 'regular'
            vis_data['nodes'].append({
                'id': node_id,
                'position': node.position,
                'type': node_type,
                'activation': node.activation,
                'resource_level': node.resource_level,
                'energy': node.energy,
                'age': node.age,
            })
        
        # Add connection data
        for source_id, node in self.nodes.items():
            for target_id, strength in node.connections.items():
                if target_id in self.nodes:  # Ensure target exists
                    vis_data['connections'].append({
                        'source': source_id,
                        'target': target_id,
                        'strength': strength,
                    })
        
        # Save to file if specified
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(vis_data, f, indent=2)
        
        return vis_data
    
    def get_network_statistics(self) -> Dict:
        """
        Calculate statistics about the network structure and performance.
        
        Returns:
            Dictionary of network statistics
        """
        stats = {
            'node_count': len(self.nodes),
            'input_nodes': len(self.input_nodes),
            'output_nodes': len(self.output_nodes),
            'regular_nodes': len(self.regular_nodes),
            'connection_count': sum(len(node.connections) for node in self.nodes.values()),
            'avg_connections_per_node': sum(len(node.connections) for node in self.nodes.values()) / max(1, len(self.nodes)),
            'total_resources': self.total_resources,
            'iteration': self.iteration,
        }
        
        # Calculate average resource levels
        if self.nodes:
            stats['avg_resource_level'] = sum(node.resource_level for node in self.nodes.values()) / len(self.nodes)
            stats['avg_energy'] = sum(node.energy for node in self.nodes.values()) / len(self.nodes)
            stats['avg_age'] = sum(node.age for node in self.nodes.values()) / len(self.nodes)
        
        # Calculate connectivity metrics
        if self.regular_nodes:
            # Average path length between nodes (approximate)
            path_lengths = self._calculate_average_path_length()
            stats['avg_path_length'] = path_lengths
        
        return stats
    
    def _calculate_average_path_length(self) -> float:
        """
        Calculate the average shortest path length between nodes.
        Uses breadth-first search for a sample of node pairs.
        
        Returns:
            Average path length
        """
        # Sample pairs of nodes (to avoid O(n²) computation for large networks)
        sample_size = min(20, len(self.regular_nodes))
        sampled_nodes = random.sample(self.regular_nodes, sample_size)
        
        path_lengths = []
        for source_id in sampled_nodes:
            for target_id in sampled_nodes:
                if source_id != target_id:
                    path_length = self._shortest_path_length(source_id, target_id)
                    if path_length > 0:  # Only count connected pairs
                        path_lengths.append(path_length)
        
        return sum(path_lengths) / max(1, len(path_lengths)) if path_lengths else float('inf')
    
    def _shortest_path_length(self, source_id: int, target_id: int) -> int:
        """
        Calculate shortest path length between two nodes using BFS.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Path length or -1 if no path exists
        """
        visited = set([source_id])
        queue = deque([(source_id, 0)])  # (node_id, distance)
        
        while queue:
            node_id, distance = queue.popleft()
            
            if node_id == target_id:
                return distance
            
            # Check all outgoing connections
            node = self.nodes[node_id]
            for next_id in node.connections:
                if next_id not in visited and next_id in self.nodes:
                    visited.add(next_id)
                    queue.append((next_id, distance + 1))
        
        return -1  # No path found
