#!/usr/bin/env python3
"""
Machine Learning Integration Demo for Enhanced Mycelium Network

This script demonstrates the machine learning capabilities of the enhanced
mycelium network, including reinforcement learning and transfer learning.
"""

import os
import sys
import random
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.reinforcement import ReinforcementLearner
from mycelium.enhanced.ml.transfer import TransferNetwork


def print_separator(title):
    """Print a section separator with title."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def demo_reinforcement_learning():
    """Demonstrate reinforcement learning for optimizing network growth."""
    print_separator("Reinforcement Learning Optimization")
    
    # Create environment
    environment = RichEnvironment(dimensions=2, size=1.0)
    print(f"Created environment: {environment.name}")
    
    # Add resources
    print("Adding resources to environment...")
    resources = [
        ((0.3, 0.7), 0.15, ResourceType.CARBON, 1.0),
        ((0.7, 0.3), 0.15, ResourceType.WATER, 1.0),
        ((0.5, 0.5), 0.1, ResourceType.NITROGEN, 0.5)
    ]
    
    for center, radius, resource_type, amount in resources:
        environment.add_nutrient_cluster(center, radius, resource_type, amount)
    
    # Create network to optimize
    network = AdaptiveMyceliumNetwork(
        environment=environment,
        input_size=3,
        output_size=1,
        initial_nodes=5
    )
    print(f"Created network with {len(network.nodes)} nodes")
    
    # Define state space and action space
    state_size = 7   # Node count, resources, environmental factors
    action_size = 6  # Different growth strategies
    
    # Create reinforcement learner
    rl_agent = ReinforcementLearner(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99
    )
    print(f"Created reinforcement learning agent with {state_size}-dimensional state space")
    print(f"Action space has {action_size} actions representing different growth strategies")
    
    # Define environment step function for RL
    def environment_step(action=None, reset=False):
        """Environment step function for RL agent."""
        if reset:
            # Reset environment and network
            environment.update(0.1)
            for _ in range(5):
                network.forward([0.5, 0.5, 0.5])
            
            # Get initial state
            state = get_state()
            return state
        
        # Process the action
        if action is not None:
            # Action represents growth strategy
            execute_growth_strategy(action)
        
        # Update environment and network
        environment.update(0.1)
        outputs = network.forward([0.5, 0.5, 0.5])
        
        # Calculate reward based on network performance
        reward = calculate_reward()
        
        # Get next state
        next_state = get_state()
        
        # Check if episode is done
        done = len(network.nodes) >= 30 or network.total_resources <= 0
        
        return next_state, reward, done, {"outputs": outputs}
    
    def get_state():
        """Convert environment and network state to RL state vector."""
        # State components:
        # 1. Normalized node count
        node_count = len(network.nodes) / 50  # Max expected nodes
        
        # 2. Resource availability
        total_resources = network.total_resources / 10  # Normalize
        
        # 3. Environmental factors
        env_factors = environment.factors
        temperature = env_factors.temperature
        moisture = env_factors.moisture
        light = env_factors.light_level
        
        # 4. Network adaptation levels
        temp_adaptation = network.temperature_adaptation
        moisture_adaptation = network.moisture_adaptation
        
        return [
            node_count,
            total_resources,
            temperature,
            moisture,
            light,
            temp_adaptation,
            moisture_adaptation
        ]
    
    def execute_growth_strategy(action):
        """Execute a growth strategy based on the action."""
        if action == 0:
            # Balanced growth
            network.growth_rate = 0.05
            network.adaptation_rate = 0.1
        elif action == 1:
            # Rapid growth
            network.growth_rate = 0.1
            network.adaptation_rate = 0.05
        elif action == 2:
            # Adaptive growth
            network.growth_rate = 0.03
            network.adaptation_rate = 0.15
        elif action == 3:
            # Resource-focused
            network.growth_rate = 0.07
            network._prioritize_resources()
        elif action == 4:
            # Specialized growth
            network.growth_rate = 0.04
            network._prioritize_specialization()
        elif action == 5:
            # Conservative growth
            network.growth_rate = 0.02
            network.adaptation_rate = 0.2
    
    def calculate_reward():
        """Calculate reward based on network performance metrics."""
        # Reward components
        node_reward = min(1.0, len(network.nodes) / 30)
        resource_reward = min(1.0, network.total_resources / 5)
        
        # Get specialization statistics
        stats = network.get_specialization_statistics()
        adaptation_level = (
            stats['adaptation']['temperature_adaptation'] + 
            stats['adaptation']['moisture_adaptation'] +
            stats['adaptation']['drought_resistance']
        ) / 3
        
        adaptation_reward = adaptation_level
        
        # Calculate connectivity (more connected = better)
        connectivity = 0
        for node in network.nodes.values():
            connectivity += len(node.connections)
        connectivity = min(1.0, connectivity / (len(network.nodes) * 3))
        
        # Weighted reward
        reward = (
            node_reward * 0.4 +
            resource_reward * 0.3 +
            adaptation_reward * 0.2 +
            connectivity * 0.1
        )
        
        return reward
    
    # For simplicity, add these methods to the network class
    def _prioritize_resources(network):
        """Add a method to prioritize resource acquisition."""
        # This is a mock implementation for the demo
        pass
    
    def _prioritize_specialization(network):
        """Add a method to prioritize node specialization."""
        # This is a mock implementation for the demo
        pass
    
    # Add these methods to the network class
    AdaptiveMyceliumNetwork._prioritize_resources = _prioritize_resources
    AdaptiveMyceliumNetwork._prioritize_specialization = _prioritize_specialization
    
    # Train the RL agent
    print("\nTraining reinforcement learning agent...")
    print(f"{'Episode':^8} | {'Reward':^8} | {'Steps':^6} | {'Nodes':^6} | {'Exploration':^10}")
    print("-" * 60)
    
    training_data = {
        "episodes": [],
        "rewards": [],
        "node_counts": [],
        "exploration_rates": []
    }
    
    num_episodes = 20  # For demo purposes
    for episode in range(num_episodes):
        # Run one training episode
        episode_stats = rl_agent.train_episode(
            environment_step_fn=environment_step,
            max_steps=50
        )
        
        # Log progress
        print(f"{episode+1:^8} | {episode_stats['reward']:8.2f} | {episode_stats['steps']:^6} | {len(network.nodes):^6} | {rl_agent.exploration_rate:^10.2f}")
        
        # Collect data for plotting
        training_data["episodes"].append(episode + 1)
        training_data["rewards"].append(episode_stats["reward"])
        training_data["node_counts"].append(len(network.nodes))
        training_data["exploration_rates"].append(rl_agent.exploration_rate)
    
    print(f"\nTraining completed over {num_episodes} episodes")
    print(f"Final Q-table size: {len(rl_agent.q_table)} states")
    print(f"Final exploration rate: {rl_agent.exploration_rate:.2f}")
    
    # Test the optimized policy
    print("\nTesting optimized policy...")
    
    # Reset environment
    state = environment_step(reset=True)
    
    # Run test episode
    total_reward = 0
    for step in range(20):
        # Choose best action
        action = rl_agent.get_best_action(state)
        
        # Execute action
        next_state, reward, done, _ = environment_step(action)
        
        # Accumulate reward
        total_reward += reward
        
        # Move to next state
        state = next_state
        
        if done:
            break
    
    print(f"Test episode completed with total reward: {total_reward:.2f}")
    print(f"Final network has {len(network.nodes)} nodes and {network.total_resources:.2f} resources")
    
    # Create a plot
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(training_data["episodes"], training_data["rewards"], 'b-')
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Plot node counts
        plt.subplot(2, 2, 2)
        plt.plot(training_data["episodes"], training_data["node_counts"], 'r-')
        plt.title("Network Size Growth")
        plt.xlabel("Episode")
        plt.ylabel("Node Count")
        plt.grid(True)
        
        # Plot exploration rate
        plt.subplot(2, 2, 3)
        plt.plot(training_data["episodes"], training_data["exploration_rates"], 'g-')
        plt.title("Exploration Rate Decay")
        plt.xlabel("Episode")
        plt.ylabel("Exploration Rate")
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        import os
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vis_dir = os.path.join(repo_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)  # Ensure directory exists
        save_path = os.path.join(vis_dir, "rl_training_results.png")
        plt.savefig(save_path)
        print(f"\nTraining plot saved to '{save_path}'")
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    return rl_agent, network, environment


def demo_transfer_learning():
    """Demonstrate transfer learning between networks."""
    print_separator("Transfer Learning Between Networks")
    
    # Create two environments with slightly different conditions
    env1 = RichEnvironment(dimensions=2, size=1.0, name="Source Environment")
    env1.factors.temperature = 0.7  # Hot environment
    env1.factors.moisture = 0.3     # Dry environment
    
    env2 = RichEnvironment(dimensions=2, size=1.0, name="Target Environment")
    env2.factors.temperature = 0.6  # Slightly cooler
    env2.factors.moisture = 0.4     # Slightly more moist
    
    print("Created two environments with different conditions:")
    print(f"Source: Temperature={env1.factors.temperature:.1f}, Moisture={env1.factors.moisture:.1f}")
    print(f"Target: Temperature={env2.factors.temperature:.1f}, Moisture={env2.factors.moisture:.1f}")
    
    # Create two networks
    source_network = AdaptiveMyceliumNetwork(environment=env1, input_size=2, output_size=1, initial_nodes=15)
    target_network = AdaptiveMyceliumNetwork(environment=env2, input_size=2, output_size=1, initial_nodes=15)
    
    print(f"Created source network with {len(source_network.nodes)} nodes")
    print(f"Created target network with {len(target_network.nodes)} nodes")
    
    # Train the source network
    print("\nTraining source network...")
    for _ in range(20):
        # Update environment
        env1.update(0.5)
        
        # Forward pass
        source_network.forward([0.7, 0.3])
    
    # Train the target network (less)
    print("Training target network (less)...")
    for _ in range(5):
        # Update environment
        env2.update(0.5)
        
        # Forward pass
        target_network.forward([0.6, 0.4])
    
    # Get initial adaptation stats
    source_stats = source_network.get_specialization_statistics()
    target_stats = target_network.get_specialization_statistics()
    
    print("\nBefore knowledge transfer:")
    print(f"Source network temperature adaptation: {source_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Source network moisture adaptation: {source_stats['adaptation']['moisture_adaptation']:.3f}")
    print(f"Target network temperature adaptation: {target_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Target network moisture adaptation: {target_stats['adaptation']['moisture_adaptation']:.3f}")
    
    # Create transfer network
    transfer = TransferNetwork(similarity_threshold=0.5)
    
    # Calculate similarity
    similarity = transfer.calculate_similarity(source_network, target_network)
    print(f"\nNetwork similarity: {similarity:.2f}")
    
    # Transfer knowledge
    if similarity >= transfer.similarity_threshold:
        print("\nTransferring knowledge from source to target network...")
        result = transfer.transfer_knowledge(source_network, target_network, transfer_rate=0.4)
        
        print(f"Transfer successful: {result['success']}")
        print(f"Knowledge gain: {result['knowledge_gain']:.3f}")
        print(f"Effective transfer rate: {result['effective_transfer_rate']:.2f}")
        
        # Get updated stats
        target_stats_after = target_network.get_specialization_statistics()
        
        print("\nAfter knowledge transfer:")
        print(f"Target network temperature adaptation: {target_stats_after['adaptation']['temperature_adaptation']:.3f} (was {target_stats['adaptation']['temperature_adaptation']:.3f})")
        print(f"Target network moisture adaptation: {target_stats_after['adaptation']['moisture_adaptation']:.3f} (was {target_stats['adaptation']['moisture_adaptation']:.3f})")
        
        # Show specialization changes
        print("\nSpecialization changes:")
        for spec_type in source_network.specializations:
            source_count = len(source_network.specializations.get(spec_type, []))
            target_before = len(target_network.specializations.get(spec_type, []))
            target_after = len(target_network.specializations.get(spec_type, []))
            
            if target_before != target_after:
                print(f"  {spec_type}: {target_before} → {target_after} nodes")
    else:
        print(f"\nNetworks too dissimilar for knowledge transfer (threshold: {transfer.similarity_threshold})")
    
    return transfer, source_network, target_network


def main():
    """Run the machine learning integration demo."""
    print("Enhanced Mycelium Network - Machine Learning Integration Demo")
    print("==========================================================")
    
    # Demo reinforcement learning
    rl_agent, rl_network, rl_env = demo_reinforcement_learning()
    
    # Demo transfer learning
    transfer, source_network, target_network = demo_transfer_learning()
    
    print_separator("Demo Completed")
    print("\nSuccessfully demonstrated machine learning integration!")
    print("Key features demonstrated:")
    print("- Reinforcement learning for network optimization")
    print("- Q-learning for growth strategy selection")
    print("- Transfer learning between different networks")
    print("- Similarity calculation for knowledge transfer")
    print("- Adaptive knowledge transfer based on similarity")


if __name__ == "__main__":
    main()
