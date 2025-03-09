"""
Adaptive Mycelium Network implementation that works with the RichEnvironment.

This module implements a more advanced version of the mycelium network that
can adapt to complex environmental conditions, handle resource diversity,
and feature more sophisticated growth patterns.
"""

import math
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict, deque

from mycelium.network import AdvancedMyceliumNetwork
from mycelium.node import MyceliumNode, Signal
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType, Environmental_Factors


class AdaptiveMyceliumNetwork(AdvancedMyceliumNetwork):
    """
    An enhanced mycelium network that adapts to complex environments.
    
    Features:
    - Resource specialization (different node types prefer different resources)
    - Environmental adaptation (nodes adapt to temp, moisture, etc.)
    - Diverse growth strategies based on environment
    - Enhanced chemical signaling
    """
    
    def __init__(
        self,
        environment: Optional[RichEnvironment] = None,
        input_size: int = 2,
        output_size: int = 1,
        initial_nodes: int = 15
    ):
        """
        Initialize the adaptive network.
        
        Args:
            environment: Rich environment instance
            input_size: Number of input nodes
            output_size: Number of output nodes
            initial_nodes: Initial number of regular nodes
        """
        # Create environment if not provided
        if environment is None:
            environment = RichEnvironment()
            
        # Call parent initializer
        super().__init__(environment, input_size, output_size, initial_nodes)
        
        # Enhanced properties
        self.adaptation_history = []
        self.growth_events = []
        
        # Resource processing efficiency
        self.resource_efficiency = {
            ResourceType.CARBON: 1.0,
            ResourceType.WATER: 1.0,
            ResourceType.NITROGEN: 1.0,
            ResourceType.SUGAR: 1.0,
            ResourceType.PHOSPHORUS: 1.0,
            ResourceType.PROTEIN: 1.0,
            ResourceType.MINERAL: 1.0,
        }
        
        # Environmental adaptation values
        self.temperature_adaptation = 0.5  # 0-1, higher = better in hot environments
        self.moisture_adaptation = 0.5     # 0-1, higher = better in wet environments
        self.drought_resistance = 0.3      # 0-1, higher = better in dry environments
        self.stress_tolerance = 0.5        # 0-1, higher = better in toxic/extreme environments
        
        # Network specialization
        self.specializations = {}          # {specialization_type: level}
        
        # Node type distribution targets
        self.node_type_targets = {
            'input': input_size,
            'output': output_size,
            'regular': initial_nodes * 0.7,
            'storage': initial_nodes * 0.1,
            'processing': initial_nodes * 0.1,
            'sensor': initial_nodes * 0.1,
        }
        
        # Initialize with some specialized nodes
        self._initialize_specialized_nodes()
    
    def _initialize_specialized_nodes(self):
        """Initialize the network with some specialized node types."""
        # Get or create a rich environment
        env = self.environment
        rich_env = isinstance(env, RichEnvironment)
        
        # Convert some regular nodes to specialized types
        nodes_to_convert = min(len(self.regular_nodes), 3)
        
        for i in range(nodes_to_convert):
            if not self.regular_nodes:
                break
                
            # Select a node to convert
            node_id = random.choice(self.regular_nodes)
            node = self.nodes[node_id]
            
            # Determine specialization based on position
            if rich_env and env.dimensions >= 3:
                # 3D environment - specialize based on height
                z = node.position[2]
                
                if z > 0.7:  # Air layer
                    specialization = 'sensor'  # Sensors work well in air
                elif z > 0.6:  # Topsoil
                    specialization = 'processing'  # Processing works well in nutrient-rich soil
                else:  # Deeper layers
                    specialization = 'storage'  # Storage works well in stable deeper layers
            else:
                # 2D environment or not rich - random specialization
                specialization = random.choice(['storage', 'processing', 'sensor'])
            
            # Apply specialization
            node.type = specialization
            
            # Adjust node properties based on specialization
            if specialization == 'storage':
                node.resource_level = 1.5  # Higher storage capacity
                node.energy = 0.7
                node.longevity += 20  # More durable
            elif specialization == 'processing':
                node.sensitivity = 1.2  # More sensitive
                node.adaptability = 1.3  # More adaptable
            elif specialization == 'sensor':
                node.sensitivity = 1.5  # Much more sensitive
                node.energy = 0.6  # Less energy efficient
            
            # Update collections
            self.regular_nodes.remove(node_id)
            if specialization not in self.specializations:
                self.specializations[specialization] = []
            self.specializations[specialization].append(node_id)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass with environmental adaptation.
        
        Args:
            inputs: Input values
            
        Returns:
            Output values
        """
        # Get environmental conditions
        env = self.environment
        rich_env = isinstance(env, RichEnvironment)
        
        if rich_env:
            # Center position for environmental sampling
            center_pos = tuple(0.5 for _ in range(env.dimensions))
            factors = env.get_environmental_factors_at(center_pos)
            
            # Apply environmental factors to network behavior
            self._adapt_to_environment(factors)
        
        # Regular forward pass
        outputs = super().forward(inputs)
        
        # Record adaptation state
        self._record_adaptation_state()
        
        return outputs
    
    def _adapt_to_environment(self, factors: Environmental_Factors):
        """
        Adapt network to the current environmental conditions.
        
        Args:
            factors: Environmental factors
        """
        # Adjust growth rate based on moisture and temperature
        moisture_factor = self._get_moisture_factor(factors.moisture)
        temp_factor = self._get_temperature_factor(factors.temperature)
        
        # Balanced growth rate based on environmental conditions
        self.growth_rate = 0.05 * moisture_factor * temp_factor
        
        # Adjust adaptation rate based on environmental variability
        self.adaptation_rate = 0.1 * (0.5 + 0.5 * factors.oxygen)
        
        # Update resource efficiency based on environmental factors
        self._update_resource_efficiency(factors)
        
        # Gradual adaptation to environment
        self._adapt_to_temperature(factors.temperature)
        self._adapt_to_moisture(factors.moisture)
        
        # Seasonal adaptations
        if factors.season == 0:  # Spring
            self.growth_rate *= 1.2  # Faster growth in spring
        elif factors.season == 2:  # Fall
            self.growth_rate *= 0.8  # Slower growth in fall
        elif factors.season == 3:  # Winter
            self.growth_rate *= 0.6  # Much slower growth in winter
    
    def _get_moisture_factor(self, moisture: float) -> float:
        """
        Get growth factor based on moisture and network adaptation.
        
        Args:
            moisture: Environmental moisture level (0-1)
            
        Returns:
            Growth factor for moisture
        """
        # Calculate ideal moisture based on adaptation
        ideal_moisture = 0.5 + 0.3 * (self.moisture_adaptation - 0.5)
        
        # Distance from ideal (0 = ideal, 1 = farthest)
        moisture_distance = abs(moisture - ideal_moisture)
        
        # Drought resistance helps in dry conditions
        if moisture < 0.3:
            moisture_distance *= (1 - self.drought_resistance * 0.5)
        
        # Convert to factor (1 at ideal, lower as you move away)
        return max(0.3, 1 - moisture_distance)
    
    def _get_temperature_factor(self, temperature: float) -> float:
        """
        Get growth factor based on temperature and network adaptation.
        
        Args:
            temperature: Environmental temperature level (0-1)
            
        Returns:
            Growth factor for temperature
        """
        # Calculate ideal temperature based on adaptation
        ideal_temp = 0.5 + 0.3 * (self.temperature_adaptation - 0.5)
        
        # Distance from ideal (0 = ideal, 1 = farthest)
        temp_distance = abs(temperature - ideal_temp)
        
        # Stress tolerance helps in temperature extremes
        temp_distance *= (1 - self.stress_tolerance * 0.3)
        
        # Convert to factor (1 at ideal, lower as you move away)
        return max(0.3, 1 - temp_distance)
    
    def _update_resource_efficiency(self, factors: Environmental_Factors):
        """
        Update resource processing efficiency based on environmental factors.
        
        Args:
            factors: Environmental factors
        """
        # Base efficiency changes
        self.resource_efficiency[ResourceType.WATER] = 0.8 + 0.4 * self.moisture_adaptation
        self.resource_efficiency[ResourceType.CARBON] = 0.7 + 0.5 * factors.light_level
        
        # pH affects nitrogen processing
        ph_factor = 1 - abs(factors.ph - 7) / 7  # 1 at neutral pH, lower at extremes
        self.resource_efficiency[ResourceType.NITROGEN] = 0.6 + 0.8 * ph_factor
        
        # Temperature affects sugar processing
        temp_factor = 0.5 + 0.8 * factors.temperature  # Higher in warmer conditions
        self.resource_efficiency[ResourceType.SUGAR] = 0.7 + 0.6 * temp_factor
    
    def _adapt_to_temperature(self, temperature: float):
        """
        Gradually adapt to environmental temperature.
        
        Args:
            temperature: Environmental temperature (0-1)
        """
        # Gradual adaptation toward environmental conditions
        target_adaptation = temperature
        
        # Move current adaptation toward target
        adaptation_rate = 0.01 * self.adaptation_rate
        self.temperature_adaptation += (target_adaptation - self.temperature_adaptation) * adaptation_rate
    
    def _adapt_to_moisture(self, moisture: float):
        """
        Gradually adapt to environmental moisture.
        
        Args:
            moisture: Environmental moisture (0-1)
        """
        # Higher adaptation rate in extreme conditions
        adaptation_rate = 0.01 * self.adaptation_rate
        
        if moisture < 0.3:
            # In dry conditions, develop drought resistance
            target_drought = 0.7
            self.drought_resistance += (target_drought - self.drought_resistance) * adaptation_rate
        else:
            # In normal/wet conditions, adapt moisture preference
            target_adaptation = moisture
            self.moisture_adaptation += (target_adaptation - self.moisture_adaptation) * adaptation_rate
    
    def _record_adaptation_state(self):
        """Record the current adaptation state for analysis."""
        self.adaptation_history.append({
            'iteration': self.iteration,
            'growth_rate': self.growth_rate,
            'adaptation_rate': self.adaptation_rate,
            'temperature_adaptation': self.temperature_adaptation,
            'moisture_adaptation': self.moisture_adaptation,
            'drought_resistance': self.drought_resistance,
            'stress_tolerance': self.stress_tolerance,
            'node_count': len(self.nodes),
            'resource_efficiency': self.resource_efficiency.copy(),
        })
    
    def _grow_network(self) -> None:
        """Enhanced growth with adaptation to environment."""
        # Use the parent implementation if not enough resources
        if self.total_resources < 5.0:
            return
        
        # Get environmental conditions if available
        env = self.environment
        rich_env = isinstance(env, RichEnvironment)
        
        if rich_env:
            # Use rich environment to influence growth
            return self._adaptive_grow_network()
        else:
            # Fall back to standard growth
            return super()._grow_network()
    
    def _adaptive_grow_network(self) -> None:
        """
        Grow the network with adaptation to the rich environment.
        
        This version considers:
        - Environmental conditions (temperature, moisture, etc.)
        - Resource distribution
        - Node specialization
        """
        # Original growth logic with adaptation
        growth_candidates = []
        for node_id, node in self.nodes.items():
            if node_id in self.input_nodes or node_id in self.output_nodes:
                continue  # Skip input/output nodes
                
            if node.resource_level > 1.5 and node.energy > 0.7:
                growth_candidates.append(node_id)
                
        if not growth_candidates:
            return
            
        # Select a source node, favoring specialized nodes sometimes
        if random.random() < 0.3 and any(self.specializations.values()):
            # Choose from specialized nodes
            specialized_nodes = []
            for nodes in self.specializations.values():
                specialized_nodes.extend(nodes)
                
            valid_nodes = [n for n in specialized_nodes if n in growth_candidates]
            if valid_nodes:
                source_id = random.choice(valid_nodes)
            else:
                source_id = random.choice(growth_candidates)
        else:
            # Choose from any candidate
            source_id = random.choice(growth_candidates)
            
        source_node = self.nodes[source_id]
        
        # Determine node type to grow
        node_type = self._determine_node_type_to_grow(source_node)
        
        # Create a new node with adapted parameters
        env = self.environment
        rich_env = isinstance(env, RichEnvironment)
        
        # Growth direction influenced by resources and environment
        nearby_resources = {}
        if isinstance(env, RichEnvironment):
            nearby_resources = env.get_resources_in_range(source_node.position, 0.3)
        
        if nearby_resources and random.random() < 0.7:
            # Grow toward richest resource position
            best_pos = max(nearby_resources.items(), key=lambda x: sum(x[1].values()))[0]
            
            # Calculate direction vector
            direction = []
            for src, tgt in zip(source_node.position, best_pos):
                direction.append(tgt - src)
                
            # Normalize and apply random variation
            direction_length = math.sqrt(sum(d**2 for d in direction))
            if direction_length > 0:
                direction = [d/direction_length * (0.8 + 0.4*random.random()) for d in direction]
            else:
                direction = [random.uniform(-0.1, 0.1) for _ in range(len(source_node.position))]
        else:
            # Random direction with environmental influence
            if env.dimensions == 2:
                # 2D - random angle
                angle = random.uniform(0, 2 * math.pi)
                direction = [math.cos(angle), math.sin(angle)]
            else:
                # 3D - consider environmental factors
                if hasattr(env, 'factors'):
                    # More upward growth in high light
                    up_bias = 0.3 * env.factors.light_level
                    
                    # Random direction with upward bias
                    theta = random.uniform(0, 2 * math.pi)
                    phi = random.uniform(0, math.pi) * (1 - up_bias)  # Bias toward smaller phi (upward)
                    
                    x = math.sin(phi) * math.cos(theta)
                    y = math.sin(phi) * math.sin(theta)
                    z = math.cos(phi)
                    
                    direction = [x, y, z]
                else:
                    # Truly random direction
                    direction = [random.uniform(-1, 1) for _ in range(env.dimensions)]
        
        # Calculate growth distance based on environmental factors
        base_distance = random.uniform(0.05, 0.15)
        
        # Adjust distance based on environmental conditions
        if rich_env:
            # Get environmental factors at source position
            factors = env.get_environmental_factors_at(source_node.position)
            
            # Longer growth in higher moisture
            moisture_factor = 0.8 + 0.4 * factors.moisture
            
            # Temperature affects growth distance (optimal at middle temperatures)
            temp_factor = 1 - abs(factors.temperature - 0.5) * 0.4
            
            # Apply environmental factors
            base_distance *= moisture_factor * temp_factor
        
        # Calculate new position
        new_position = []
        for i, coord in enumerate(source_node.position):
            new_coord = coord + direction[i] * base_distance
            # Ensure within bounds
            new_coord = max(0, min(env.size, new_coord))
            new_position.append(new_coord)
        
        new_position = tuple(new_position)
        
        # Check if position is valid
        if not env.is_position_valid(new_position):
            return
        
        # Create new node
        new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        new_node = MyceliumNode(new_id, new_position, node_type)
        
        # Adapt new node properties to environment
        if rich_env:
            factors = env.get_environmental_factors_at(new_position)
            
            # Adjust sensitivity based on temperature
            new_node.sensitivity *= (0.8 + 0.4 * factors.temperature)
            
            # Adjust adaptability based on environmental variables
            new_node.adaptability *= (0.8 + 0.4 * self.adaptation_rate)
            
            # Adjust resource processing based on node type and environment
            if node_type == 'storage':
                # Storage nodes have higher resource capacity
                new_node.resource_level *= 1.5
                new_node.longevity += 10
            elif node_type == 'processing':
                # Processing nodes are more sensitive to input
                new_node.sensitivity *= 1.2
                new_node.adaptability *= 1.3
            elif node_type == 'sensor':
                # Sensor nodes are highly sensitive but energy inefficient
                new_node.sensitivity *= 1.5
                new_node.energy *= 0.8
        
        # Add to network
        self.nodes[new_id] = new_node
        
        # Update collections based on node type
        if node_type == 'regular':
            self.regular_nodes.append(new_id)
        else:
            # Add to specialization collection
            if node_type not in self.specializations:
                self.specializations[node_type] = []
            self.specializations[node_type].append(new_id)
        
        # Connect to source node
        source_node.connect_to(new_node, strength=0.5)
        
        # Connect to nearby nodes based on node type
        self._connect_to_nearby_nodes(new_node, node_type)
        
        # Use resources for growth
        resource_cost = 1.0
        if rich_env:
            # Apply resource efficiency
            primary_resource = ResourceType.CARBON
            if node_type == 'storage':
                primary_resource = ResourceType.SUGAR
            elif node_type == 'processing':
                primary_resource = ResourceType.NITROGEN
                
            # Adjust cost based on efficiency
            resource_cost *= (1.0 / self.resource_efficiency.get(primary_resource, 1.0))
        
        self.total_resources -= resource_cost
        source_node.resource_level -= 0.5
        source_node.energy -= 0.3
        
        # Record growth event
        self.growth_events.append({
            'iteration': self.iteration,
            'source_id': source_id,
            'new_id': new_id,
            'position': new_position,
            'node_type': node_type
        })
    
    def _determine_node_type_to_grow(self, source_node) -> str:
        """
        Determine what type of node to grow based on network needs.
        
        Args:
            source_node: The source node for growth
            
        Returns:
            Node type to create
        """
        # Count current node types
        node_counts = {
            'regular': len(self.regular_nodes),
            'input': len(self.input_nodes),
            'output': len(self.output_nodes)
        }
        
        # Count specialized nodes
        for node_type, nodes in self.specializations.items():
            node_counts[node_type] = len(nodes)
        
        # Calculate proportional distance from targets
        type_needs = {}
        for node_type, target in self.node_type_targets.items():
            if target > 0:
                current = node_counts.get(node_type, 0)
                # Proportional distance from target (higher = more needed)
                type_needs[node_type] = max(0, (target - current) / target)
        
        # Special case: always maintain minimum of regular nodes
        if node_counts.get('regular', 0) < 3:
            return 'regular'
        
        # Bias based on source node type
        source_type = source_node.type
        source_bias = {
            'regular': {'regular': 0.7, 'storage': 0.15, 'processing': 0.1, 'sensor': 0.05},
            'storage': {'regular': 0.3, 'storage': 0.5, 'processing': 0.15, 'sensor': 0.05},
            'processing': {'regular': 0.2, 'storage': 0.2, 'processing': 0.5, 'sensor': 0.1},
            'sensor': {'regular': 0.3, 'storage': 0.1, 'processing': 0.3, 'sensor': 0.3},
        }
        
        # Default bias if source type not in mapping
        default_bias = {'regular': 0.6, 'storage': 0.15, 'processing': 0.15, 'sensor': 0.1}
        
        # Get appropriate bias
        bias = source_bias.get(source_type, default_bias)
        
        # Combine needs and bias to get weighted probabilities
        weights = {}
        for node_type in ['regular', 'storage', 'processing', 'sensor']:
            need = type_needs.get(node_type, 0)
            node_bias = bias.get(node_type, 0.1)
            weights[node_type] = need * 0.7 + node_bias * 0.3
        
        # Select node type based on weights
        options = []
        probabilities = []
        for node_type, weight in weights.items():
            options.append(node_type)
            probabilities.append(weight)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Default to uniform distribution
            probabilities = [1.0 / len(options)] * len(options)
        
        # Choose node type
        return random.choices(options, probabilities)[0]
    
    def _connect_to_nearby_nodes(self, new_node, node_type):
        """
        Connect new node to nearby nodes based on node type.
        
        Args:
            new_node: The newly created node
            node_type: Type of the new node
        """
        # Determine connection parameters based on node type
        if node_type == 'storage':
            # Storage nodes connect to fewer nodes but with stronger connections
            max_connections = 3
            connection_probability = 0.2
            connection_strength = 0.7
        elif node_type == 'processing':
            # Processing nodes connect to many nodes
            max_connections = 6
            connection_probability = 0.4
            connection_strength = 0.5
        elif node_type == 'sensor':
            # Sensor nodes connect to specific nodes
            max_connections = 2
            connection_probability = 0.3
            connection_strength = 0.8
        else:
            # Regular nodes - default behavior
            max_connections = 4
            connection_probability = 0.3
            connection_strength = 0.4
        
        # Find potential connections
        connections_made = 0
        
        # Prioritize connections based on node type
        if node_type == 'sensor':
            # Sensors prioritize connecting to processing nodes
            priority_nodes = self.specializations.get('processing', [])
            for target_id in priority_nodes:
                if connections_made >= max_connections:
                    break
                    
                target_node = self.nodes[target_id]
                if new_node.can_connect_to(target_node, self.environment, max_distance=0.2):
                    if random.random() < connection_probability * 1.5:  # Higher probability for priority
                        new_node.connect_to(target_node, connection_strength)
                        connections_made += 1
        
        # Connect to other nodes based on proximity
        for node_id, node in self.nodes.items():
            if node_id == new_node.id:
                continue  # Skip self
                
            if connections_made >= max_connections:
                break
                
            if new_node.can_connect_to(node, self.environment, max_distance=0.15):
                if random.random() < connection_probability:
                    new_node.connect_to(node, connection_strength)
                    connections_made += 1
    
    def get_specialization_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about node specializations.
        
        Returns:
            Dictionary with specialization statistics
        """
        stats = {
            'node_counts': {
                'total': len(self.nodes),
                'input': len(self.input_nodes),
                'output': len(self.output_nodes),
                'regular': len(self.regular_nodes)
            },
            'specializations': {},
            'adaptation': {
                'temperature_adaptation': self.temperature_adaptation,
                'moisture_adaptation': self.moisture_adaptation,
                'drought_resistance': self.drought_resistance,
                'stress_tolerance': self.stress_tolerance
            },
            'resource_efficiency': {k.name: v for k, v in self.resource_efficiency.items()}
        }
        
        # Add specialized node counts
        for spec_type, nodes in self.specializations.items():
            stats['node_counts'][spec_type] = len(nodes)
            
            # Calculate average properties for this specialization
            if nodes:
                sensitivities = [self.nodes[node_id].sensitivity for node_id in nodes]
                adaptability = [self.nodes[node_id].adaptability for node_id in nodes]
                resource_levels = [self.nodes[node_id].resource_level for node_id in nodes]
                energy_levels = [self.nodes[node_id].energy for node_id in nodes]
                
                stats['specializations'][spec_type] = {
                    'avg_sensitivity': sum(sensitivities) / len(sensitivities),
                    'avg_adaptability': sum(adaptability) / len(adaptability),
                    'avg_resource_level': sum(resource_levels) / len(resource_levels),
                    'avg_energy': sum(energy_levels) / len(energy_levels)
                }
        
        return stats

    def visualize_network_data(self) -> Dict[str, Any]:
        """
        Generate data for visualizing the network with specializations.
        
        Returns:
            Dictionary with visualization data
        """
        # Get basic visualization data
        vis_data = super().visualize_network()
        
        # Enhance with specialization information
        for node_data in vis_data['nodes']:
            node_id = node_data['id']
            
            # Determine specialization category
            if node_id in self.input_nodes:
                category = 'input'
            elif node_id in self.output_nodes:
                category = 'output'
            elif node_id in self.regular_nodes:
                category = 'regular'
            else:
                # Find in specializations
                category = 'regular'  # Default
                for spec_type, nodes in self.specializations.items():
                    if node_id in nodes:
                        category = spec_type
                        break
            
            # Add category to node data
            node_data['category'] = category
            
            # Add specialized properties
            node = self.nodes[node_id]
            node_data['sensitivity'] = node.sensitivity
            node_data['adaptability'] = node.adaptability
        
        # Add adaptation data
        vis_data['adaptation'] = {
            'temperature': self.temperature_adaptation,
            'moisture': self.moisture_adaptation,
            'drought_resistance': self.drought_resistance,
            'stress_tolerance': self.stress_tolerance,
            'resource_efficiency': {k.name: v for k, v in self.resource_efficiency.items()}
        }
        
        return vis_data
