"""
Batch-optimized mycelium network implementation.

This module provides an enhanced version of the mycelium network
that uses batch processing for improved performance.
"""

import math
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Union, Callable

from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.network import AdvancedMyceliumNetwork
from mycelium.node import MyceliumNode, Signal


class BatchProcessingNetwork(AdaptiveMyceliumNetwork):
    """
    An optimized mycelium network that uses batch processing for performance.
    
    Key optimizations:
    - Batch processing of signals rather than individual handling
    - Cached distance calculations
    - Optimized resource allocation
    - Parallel-friendly operations
    """
    
    def __init__(
        self,
        environment=None,
        input_size: int = 2,
        output_size: int = 1,
        initial_nodes: int = 15,
        batch_size: int = 10
    ):
        """
        Initialize the batch processing network.
        
        Args:
            environment: Environment instance
            input_size: Number of input nodes
            output_size: Number of output nodes
            initial_nodes: Initial number of regular nodes
            batch_size: Signal batch size for processing
        """
        # Call parent initializer for basic setup
        AdvancedMyceliumNetwork.__init__(self, environment, input_size, output_size)
        
        # Initialize parent class attributes that might not be set
        self.growth_rate = 0.1
        self.prune_rate = 0.05
        self.total_resources = 1.0
        self.iteration = 0
        self.active_signals = []
        
        # Create initial network structure
        self.regular_nodes = []
        self.specializations = {}
        self._create_initial_structure(initial_nodes)
        
        # Batch processing parameters
        self.batch_size = batch_size
        
        # Signal batching
        self.signal_batches = defaultdict(list)  # {target_id: [signals]}
        
        # Distance cache for frequently used calculations
        self.distance_cache = {}  # {(node1_id, node2_id): distance}
        
        # Cached node properties for quick access
        self.node_cache = {}  # {node_id: {property: value}}
        
        # Initialize caches
        self._initialize_caches()
    
    def _initialize_caches(self):
        """Initialize caches for faster operation."""
        # Cache node properties
        for node_id, node in self.nodes.items():
            self.node_cache[node_id] = {
                'position': node.position,
                'type': node.type,
                'sensitivity': node.sensitivity,
                'adaptability': node.adaptability
            }
        
        # Pre-calculate commonly used distances
        self._update_distance_cache()
    
    def _update_distance_cache(self):
        """Update the distance cache for node pairs."""
        # Limit cache size for memory efficiency
        max_cache_entries = 1000
        
        # Clear cache if too large
        if len(self.distance_cache) > max_cache_entries:
            self.distance_cache.clear()
        
        # Calculate distances for frequently used pairs
        # Focus on connections and nearby nodes
        for source_id, node in self.nodes.items():
            source_pos = node.position
            
            # Cache distances to connected nodes
            for target_id in node.connections.keys():
                if target_id in self.nodes and (source_id, target_id) not in self.distance_cache:
                    target_pos = self.nodes[target_id].position
                    distance = self.environment.calculate_distance(source_pos, target_pos)
                    self.distance_cache[(source_id, target_id)] = distance
                    self.distance_cache[(target_id, source_id)] = distance  # Symmetric
    
    def get_cached_distance(self, node1_id, node2_id):
        """
        Get cached distance between two nodes or calculate if not cached.
        
        Args:
            node1_id, node2_id: Node IDs
            
        Returns:
            Distance between nodes
        """
        # Check cache first
        cache_key = (node1_id, node2_id)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Calculate and cache if not found
        if node1_id in self.nodes and node2_id in self.nodes:
            pos1 = self.nodes[node1_id].position
            pos2 = self.nodes[node2_id].position
            distance = self.environment.calculate_distance(pos1, pos2)
            
            # Cache the result
            self.distance_cache[cache_key] = distance
            self.distance_cache[(node2_id, node1_id)] = distance  # Symmetric
            
            return distance
        
        # Default if nodes don't exist
        return float('inf')
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Optimized forward pass using batch processing.
        
        Args:
            inputs: Input values
            
        Returns:
            Output values
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Clear previous activations and prepare signal batches
        for node in self.nodes.values():
            node.activation = 0.0
        
        self.active_signals = []
        self.signal_batches.clear()
        
        # Feed inputs to input nodes and create initial signals
        for i, input_value in enumerate(inputs):
            input_id = self.input_nodes[i]
            self.nodes[input_id].activation = input_value
            
            # Create nutrient signals based on input strength
            if input_value > 0.1:
                signal = self.nodes[input_id].emit_signal(
                    'nutrient', 
                    input_value, 
                    {'source': 'input'}
                )
                # Add to active signals and batches
                self.active_signals.append((input_id, signal))
                
                # Batch signals by target (all connected nodes)
                for target_id in self.nodes[input_id].connections:
                    if target_id in self.nodes:
                        self.signal_batches[target_id].append((input_id, signal))
        
        # Process activation through the network using batched signals
        visited_nodes = set(self.input_nodes)
        propagation_queue = deque([(input_id, 0) for input_id in self.input_nodes])
        
        # Breadth-first propagation of activation
        while propagation_queue:
            batch_size = min(self.batch_size, len(propagation_queue))
            current_batch = [propagation_queue.popleft() for _ in range(batch_size)]
            
            # Process batch of nodes
            for node_id, depth in current_batch:
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
                        # Add to active signals list
                        self.active_signals.append((node_id, signal))
                        
                        # Add to batched signals
                        self.signal_batches[target_id].append((node_id, signal))
        
        # Process batched signals
        self._process_batched_signals()
        
        # Process hidden nodes - apply activation function
        for node_id in self.regular_nodes:
            self.nodes[node_id].process_signal(self.nodes[node_id].activation)
        
        # Process specialized nodes if any
        if hasattr(self, 'specializations'):
            for nodes in self.specializations.values():
                for node_id in nodes:
                    if node_id in self.nodes:
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
        
        return outputs
    
    def _process_batched_signals(self):
        """Process all signals in batches for better performance."""
        # Process each batch of signals going to the same target
        for target_id, signals in self.signal_batches.items():
            if target_id not in self.nodes:
                continue
                
            target_node = self.nodes[target_id]
            
            # Group signals by type
            grouped_signals = defaultdict(list)
            for source_id, signal in signals:
                grouped_signals[signal.type].append((source_id, signal))
            
            # Process each signal type in batch
            for signal_type, type_signals in grouped_signals.items():
                self._process_signal_batch(target_node, signal_type, type_signals)
    
    def _process_signal_batch(self, target_node, signal_type, signals):
        """
        Process a batch of signals of the same type.
        
        Args:
            target_node: Target node receiving signals
            signal_type: Type of signals
            signals: List of (source_id, signal) tuples
        """
        # Skip if no signals
        if not signals:
            return
            
        # Calculate combined effect based on signal type
        if signal_type == 'nutrient':
            # Nutrients increase resources
            total_effect = sum(signal.strength for _, signal in signals)
            # Apply diminishing returns for large batches
            if len(signals) > 5:
                total_effect = total_effect * (1 - 0.01 * (len(signals) - 5))
                
            # Apply resource increase
            target_node.resource_level = min(2.0, target_node.resource_level + total_effect * 0.2)
            
        elif signal_type == 'danger':
            # Danger increases sensitivity
            max_effect = max(signal.strength for _, signal in signals)
            # Take the maximum effect for danger signals
            target_node.sensitivity = min(2.0, target_node.sensitivity + max_effect * 0.1)
            
        elif signal_type == 'reinforcement':
            # Process each reinforcement signal individually
            # These often affect specific connections
            for source_id, signal in signals:
                if 'connection_id' in signal.metadata:
                    conn_id = signal.metadata['connection_id']
                    if conn_id in target_node.connections:
                        target_node.connections[conn_id] *= (1 + signal.strength * 0.05)
        
        # Record signals received for node adaptations
        for _, signal in signals:
            target_node.received_signals.append(signal)
    
    def _allocate_resources(self):
        """Optimized batch resource allocation algorithm."""
        # Calculate activity level for each node
        activity_levels = {}
        total_activity = 0
        
        for node_id, node in self.nodes.items():
            # Activity based on activation and signal processing
            activity = node.activation + 0.1 * len(node.received_signals)
            activity_levels[node_id] = activity
            total_activity += activity
        
        # Fast-path for low activity
        if total_activity < 0.1:
            return
        
        # Batch resource distribution
        resources_per_node = {}
        chunk_size = self.total_resources * 0.1  # 10% of resources in one batch
        
        # Normalize activity levels
        for node_id, activity in activity_levels.items():
            proportion = activity / total_activity
            resources_per_node[node_id] = chunk_size * proportion
        
        # Allocate to each node in a batch
        for node_id, amount in resources_per_node.items():
            self.nodes[node_id].allocate_resources(amount)
    
    def train(self, 
              inputs: List[List[float]], 
              targets: List[List[float]], 
              epochs: int = 10, 
              learning_rate: float = 0.1) -> List[float]:
        """
        Optimized training using batch processing.
        
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
            batch_size = min(self.batch_size, len(inputs))
            epoch_error = 0.0
            
            # Create training pairs and shuffle
            training_data = list(zip(inputs, targets))
            random.shuffle(training_data)
            
            # Process in mini-batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_inputs = [b[0] for b in batch]
                batch_targets = [b[1] for b in batch]
                
                # Process mini-batch
                batch_error = self._train_batch(batch_inputs, batch_targets, learning_rate)
                epoch_error += batch_error
            
            # Calculate average error for this epoch
            avg_error = epoch_error / len(inputs)
            epoch_errors.append(avg_error)
            
            # Adapt learning rate based on progress
            if epoch > 0 and avg_error > epoch_errors[-2]:
                learning_rate *= 0.8  # Reduce learning rate if error increases
            elif epoch > 0 and avg_error < 0.8 * epoch_errors[-2]:
                learning_rate *= 1.05  # Slightly increase if error decreases significantly
        
        return epoch_errors
    
    def _get_incoming_connections(self, node_id):
        """
        Get all incoming connections to a node.
        
        Args:
            node_id: ID of the node to get connections for
            
        Returns:
            Dictionary mapping source node IDs to connection strengths
        """
        incoming = {}
        for source_id, node in self.nodes.items():
            # If this node connects to the target
            if node_id in node.connections:
                incoming[source_id] = node.connections[node_id]
        return incoming
    
    def _train_batch(self, batch_inputs, batch_targets, learning_rate):
        """
        Train on a batch of inputs/targets.
        
        Args:
            batch_inputs: List of input vectors for the batch
            batch_targets: List of target vectors for the batch
            learning_rate: Learning rate for updates
            
        Returns:
            Total error for the batch
        """
        batch_error = 0.0
        weight_updates = defaultdict(float)  # {(source_id, target_id): update}
        
        # Forward pass and error calculation for all examples in batch
        for input_vec, target_vec in zip(batch_inputs, batch_targets):
            # Forward pass
            outputs = self.forward(input_vec)
            
            # Calculate error
            errors = [t - o for t, o in zip(target_vec, outputs)]
            sample_error = sum(e**2 for e in errors) / len(errors)
            batch_error += sample_error
            
            # Calculate weight updates for output layer
            for i, error in enumerate(errors):
                output_id = self.output_nodes[i]
                
                # Calculate updates for connections to this output
                for source_id, strength in self._get_incoming_connections(output_id).items():
                    source_node = self.nodes[source_id]
                    
                    # Calculate weight update
                    update = learning_rate * error * source_node.activation
                    
                    # Accumulate update (will be applied at end of batch)
                    weight_updates[(source_id, output_id)] += update
        
        # Apply accumulated weight updates
        for (source_id, target_id), update in weight_updates.items():
            # Get current weight
            if source_id not in self.nodes or target_id not in self.nodes[source_id].connections:
                continue
                
            current_weight = self.nodes[source_id].connections[target_id]
            
            # Calculate average update (divide by batch size)
            avg_update = update / len(batch_inputs)
            
            # Apply update with bounds
            new_weight = max(0.01, min(2.0, current_weight + avg_update))
            self.nodes[source_id].connections[target_id] = new_weight
        
        return batch_error
        
    def _create_initial_structure(self, initial_nodes):
        """Create the initial network structure with the specified number of nodes.
        
        Args:
            initial_nodes: Number of regular nodes to create
        """
        # Initialize node ID counter
        next_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        
        # Create regular nodes
        for i in range(initial_nodes):
            # Create position - distribute nodes evenly in hidden layer
            layer_pos = 0.5  # Hidden layer is at 0.5
            angle = 2 * math.pi * i / initial_nodes
            distance = random.uniform(0.1, 0.4)  # Random distance from center
            x = 0.5 + distance * math.cos(angle)
            y = 0.5 + distance * math.sin(angle)
            
            position = (x, y) if self.environment.dimensions == 2 else (x, y, layer_pos)
            
            # Create new node
            from mycelium.node import MyceliumNode
            node = MyceliumNode(next_id, position, "regular")
            
            # Add to network
            self.nodes[next_id] = node
            self.regular_nodes.append(next_id)
            
            # Create some initial connections
            self._create_initial_connections(next_id)
            
            next_id += 1
    
    def _create_initial_connections(self, node_id):
        """Create initial connections for a new node.
        
        Args:
            node_id: ID of the node to create connections for
        """
        # Connect to input nodes
        for input_id in self.input_nodes:
            if random.random() < 0.5:
                self.nodes[input_id].connections[node_id] = random.uniform(0.1, 1.0)
        
        # Connect to output nodes
        for output_id in self.output_nodes:
            if random.random() < 0.5:
                self.nodes[node_id].connections[output_id] = random.uniform(0.1, 1.0)
        
        # Connect to some existing regular nodes
        for other_id in self.regular_nodes:
            if other_id != node_id and random.random() < 0.3:
                self.nodes[node_id].connections[other_id] = random.uniform(0.1, 1.0)
                
    def _update_caches(self):
        """Update node property caches."""
        # Update changed node properties
        for node_id, node in self.nodes.items():
            # Skip if node already in cache
            if node_id in self.node_cache:
                # Update only if changed
                if (self.node_cache[node_id]['sensitivity'] != node.sensitivity or
                    self.node_cache[node_id]['adaptability'] != node.adaptability):
                    
                    self.node_cache[node_id]['sensitivity'] = node.sensitivity
                    self.node_cache[node_id]['adaptability'] = node.adaptability
            else:
                # Add new node to cache
                self.node_cache[node_id] = {
                    'position': node.position,
                    'type': node.type,
                    'sensitivity': node.sensitivity,
                    'adaptability': node.adaptability
                }
    
    def _update_network_state(self):
        """Update network state with optimized batch operations."""
        # Update caches
        self._update_caches()
        
        # Update distance cache if needed
        if self.iteration % 10 == 0:  # Only update periodically
            self._update_distance_cache()
