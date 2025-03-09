#!/usr/bin/env python3
"""
Environment Adaptation Example with Mycelium Network

This example demonstrates how mycelium networks can adapt to changes
in their environment, including resource distribution, obstacle placement,
and environmental conditions like moisture and temperature.
"""

import os
import sys
import random
import math
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from mycelium import Environment, AdvancedMyceliumNetwork


class EnhancedEnvironment(Environment):
    """Environment with additional properties for demonstrating adaptation."""
    
    def __init__(self, dimensions: int = 2, size: float = 1.0):
        """Initialize the enhanced environment."""
        super().__init__(dimensions, size)
        
        # Environmental conditions
        self.moisture_level = 0.7  # Water content (0.0 to 1.0)
        self.temperature = 0.5     # Normalized temperature (0.0 to 1.0)
        self.nutrient_richness = 0.6  # Overall nutrient level (0.0 to 1.0)
        
        # Nutrient clusters
        self.nutrient_clusters = []  # [(center, radius, type, amount),...]
    
    def add_nutrient_cluster(self, 
                           center: Tuple[float, ...], 
                           radius: float, 
                           nutrient_type: str,
                           amount: float):
        """Add a cluster of nutrients to the environment."""
        self.nutrient_clusters.append((center, radius, nutrient_type, amount))
        
        # Add resources based on the cluster
        n_resources = int(radius * 20)  # More resources for larger clusters
        for _ in range(n_resources):
            # Random position within the cluster
            pos = []
            for dim in range(self.dimensions):
                offset = random.uniform(-radius, radius)
                p = center[dim] + offset
                p = max(0, min(self.size, p))
                pos.append(p)
            
            position = tuple(pos)
            
            # Calculate amount based on distance from center
            distance = self.calculate_distance(center, position)
            if distance <= radius:
                # Resources decrease with distance from center
                resource_amount = amount * (1 - distance / radius)
                self.add_resource(position, resource_amount)
    
    def update_conditions(self, delta_time: float = 0.1):
        """Update environmental conditions over time."""
        # Random small fluctuations
        self.moisture_level += random.uniform(-0.05, 0.05) * delta_time
        self.moisture_level = max(0.1, min(0.9, self.moisture_level))
        
        self.temperature += random.uniform(-0.03, 0.03) * delta_time
        self.temperature = max(0.2, min(0.8, self.temperature))
        
        # Nutrient richness decreases over time as resources are consumed
        self.nutrient_richness = max(0.1, self.nutrient_richness - 0.01 * delta_time)
        
        # Decay resources slightly
        for pos in list(self.resources.keys()):
            self.resources[pos] *= (1 - 0.02 * delta_time)
            if self.resources[pos] < 0.05:
                del self.resources[pos]


class AdaptiveMyceliumNetwork(AdvancedMyceliumNetwork):
    """Extended mycelium network with enhanced adaptation abilities."""
    
    def __init__(self, 
               environment: EnhancedEnvironment = None,
               input_size: int = 2, 
               output_size: int = 1, 
               initial_nodes: int = 15):
        """Initialize the adaptive network."""
        # Create environment if not provided
        if environment is None:
            environment = EnhancedEnvironment()
            
        super().__init__(environment, input_size, output_size, initial_nodes)
        
        # Adaptation statistics
        self.growth_events = []
        self.adaptation_history = []
        self.resource_efficiency = 1.0
        self.stress_tolerance = 0.5
        
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass with environmental adaptation."""
        # Update environment
        if isinstance(self.environment, EnhancedEnvironment):
            self.environment.update_conditions()
        
        # Regular forward pass
        outputs = super().forward(inputs)
        
        # Adapt to environment
        self._adapt_to_environment()
        
        return outputs
    
    def _adapt_to_environment(self):
        """Adapt network to the current environmental conditions."""
        env = self.environment
        if not isinstance(env, EnhancedEnvironment):
            return
            
        # Adjust growth rate based on moisture
        self.growth_rate = 0.05 * (0.5 + env.moisture_level)
        
        # Adjust adaptation rate based on temperature
        self.adaptation_rate = 0.1 * (0.5 + env.temperature)
        
        # Resource efficiency change based on nutrient richness
        target_efficiency = 1.0 - 0.5 * env.nutrient_richness
        self.resource_efficiency += (target_efficiency - self.resource_efficiency) * 0.05
        
        # Record adaptation
        self.adaptation_history.append({
            'iteration': self.iteration,
            'growth_rate': self.growth_rate,
            'adaptation_rate': self.adaptation_rate,
            'resource_efficiency': self.resource_efficiency,
            'moisture': env.moisture_level,
            'temperature': env.temperature,
            'nutrients': env.nutrient_richness
        })
    
    def _grow_network(self) -> None:
        """Enhanced growth with adaptation to environment."""
        # Check resource levels first
        if self.total_resources < 5.0:
            return
            
        # Original growth logic with adaptation
        growth_candidates = []
        for node_id, node in self.nodes.items():
            if node.resource_level > 1.5 and node.energy > 0.7:
                growth_candidates.append(node_id)
                
        if not growth_candidates:
            return
            
        # Select a source node
        source_id = random.choice(growth_candidates)
        source_node = self.nodes[source_id]
        
        # Create a new node with adapted parameters
        env = self.environment
        
        # Direction influenced by resources
        nearby_resources = []
        if isinstance(env, EnhancedEnvironment):
            nearby_resources = env.get_resources_in_range(source_node.position, 0.3)
        
        if nearby_resources:
            # Grow toward resources
            resource_pos = max(nearby_resources.items(), key=lambda x: x[1])[0]
            angle = math.atan2(
                resource_pos[1] - source_node.position[1],
                resource_pos[0] - source_node.position[0]
            )
        else:
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
        
        # Distance influenced by environmental factors
        distance = random.uniform(0.05, 0.15)
        if isinstance(env, EnhancedEnvironment):
            # Longer growth in higher moisture
            distance *= (0.8 + 0.4 * env.moisture_level)
        
        # Calculate new position
        new_x = source_node.position[0] + distance * math.cos(angle)
        new_y = source_node.position[1] + distance * math.sin(angle)
        new_position = (
            max(0, min(1, new_x)),
            max(0, min(1, new_y))
        )
        
        # Check if position is valid
        if not env.is_position_valid(new_position):
            return
        
        # Create new node
        new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        new_node = MyceliumNode(new_id, new_position, 'regular')
        
        # Adapt new node properties to environment
        if isinstance(env, EnhancedEnvironment):
            # More sensitive in warmer environments
            new_node.sensitivity *= (0.8 + 0.4 * env.temperature)
            
            # Higher adaptability in variable environments
            new_node.adaptability *= (0.8 + 0.4 * self.adaptation_rate)
            
            # Lower metabolism in nutrient-poor environments
            new_node.energy *= self.resource_efficiency
        
        # Add to network
        self.nodes[new_id] = new_node
        self.regular_nodes.append(new_id)
        
        # Connect to source node
        source_node.connect_to(new_node, strength=0.5)
        
        # Connect to nearby nodes
        for node_id, node in self.nodes.items():
            if node_id != new_id and node_id != source_id:
                if new_node.can_connect_to(node, env, max_distance=0.15):
                    if random.random() < 0.3:
                        new_node.connect_to(node)
        
        # Use resources
        resource_cost = 1.0 * self.resource_efficiency
        self.total_resources -= resource_cost
        source_node.resource_level -= 0.5 * self.resource_efficiency
        source_node.energy -= 0.3 * self.resource_efficiency
        
        # Record growth event
        self.growth_events.append({
            'iteration': self.iteration,
            'source_id': source_id,
            'new_id': new_id,
            'position': new_position
        })


def main():
    """Run the environment adaptation example."""
    print("Environment Adaptation Example")
    print("=============================")
    
    # Create enhanced environment
    env = EnhancedEnvironment(dimensions=2)
    
    # Add resource clusters
    print("\nCreating environment with resource clusters...")
    env.add_nutrient_cluster((0.3, 0.3), 0.15, "carbon", 1.2)
    env.add_nutrient_cluster((0.7, 0.7), 0.12, "nitrogen", 1.0)
    env.add_nutrient_cluster((0.5, 0.2), 0.10, "phosphorus", 0.8)
    
    # Add obstacles
    env.add_obstacle((0.5, 0.5), 0.1)
    env.add_obstacle((0.2, 0.7), 0.08)
    
    print(f"Environment created with:")
    print(f"- {len(env.resources)} resource points")
    print(f"- {len(env.obstacles)} obstacles")
    print(f"- Moisture: {env.moisture_level:.2f}")
    print(f"- Temperature: {env.temperature:.2f}")
    print(f"- Nutrient richness: {env.nutrient_richness:.2f}")
    
    # Create adaptive network
    print("\nInitializing adaptive mycelium network...")
    network = AdaptiveMyceliumNetwork(
        environment=env,
        input_size=2,
        output_size=1,
        initial_nodes=15
    )
    
    # Run simulation for multiple iterations
    print("\nRunning network simulation with environmental adaptation...")
    iterations = 30
    
    for i in range(iterations):
        # Create random input
        inputs = [random.random() for _ in range(2)]
        
        # Forward pass
        outputs = network.forward(inputs)
        
        # Report statistics periodically
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\nIteration {i+1}:")
            print(f"- Nodes: {len(network.nodes)}")
            print(f"- Connections: {sum(len(node.connections) for node in network.nodes.values())}")
            print(f"- Resources: {network.total_resources:.2f}")
            print(f"- Growth rate: {network.growth_rate:.3f}")
            print(f"- Adaptation rate: {network.adaptation_rate:.3f}")
            print(f"- Resource efficiency: {network.resource_efficiency:.3f}")
            
            # Environment conditions
            print(f"- Environment - Moisture: {env.moisture_level:.2f}, Temperature: {env.temperature:.2f}")
    
    # Show growth statistics
    print(f"\nTotal growth events: {len(network.growth_events)}")
    
    # Show adaptation statistics
    print("\nAdaptation over time:")
    for i, adaptation in enumerate(network.adaptation_history[::5]):  # Show every 5th record
        print(f"Iteration {adaptation['iteration']}: " +
              f"Growth rate: {adaptation['growth_rate']:.3f}, " +
              f"Resource efficiency: {adaptation['resource_efficiency']:.3f}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    # Import here to avoid circular imports
    from mycelium.node import MyceliumNode
    main()
