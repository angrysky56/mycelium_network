#!/usr/bin/env python3
"""
Performance profiling tool for the mycelium network.

This script profiles the performance of different components of the
mycelium network to identify bottlenecks and optimization opportunities.
"""

import os
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.resource import ResourceType


def profile_environment_creation(num_iterations=5):
    """Profile the environment creation process."""
    print(f"Profiling environment creation ({num_iterations} iterations)...")
    
    def create_environments():
        environments = []
        for _ in range(num_iterations):
            env = RichEnvironment(dimensions=3, size=1.0)
            # Add resources
            for i in range(10):
                x, y, z = 0.1 * i + 0.05, 0.1 * i + 0.05, 0.5
                env.add_nutrient_cluster((x, y, z), 0.2, ResourceType.CARBON, 1.0)
                env.add_nutrient_cluster((x + 0.05, y + 0.05, z), 0.15, ResourceType.WATER, 0.8)
            environments.append(env)
        return environments
    
    pr = cProfile.Profile()
    pr.enable()
    environments = create_environments()
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 time-consuming functions
    print(s.getvalue())
    
    return environments


def profile_environment_update(environments, num_steps=100):
    """Profile the environment update process."""
    print(f"Profiling environment update ({num_steps} steps)...")
    
    env = environments[0]
    
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(num_steps):
        env.update(delta_time=0.1)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_network_creation(environments, num_networks=5):
    """Profile the network creation process."""
    print(f"Profiling network creation ({num_networks} networks)...")
    
    def create_networks():
        networks = []
        for i in range(min(num_networks, len(environments))):
            network = AdaptiveMyceliumNetwork(
                environment=environments[i],
                input_size=5,
                output_size=3,
                initial_nodes=50  # Increase to stress test
            )
            networks.append(network)
        return networks
    
    pr = cProfile.Profile()
    pr.enable()
    networks = create_networks()
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    return networks


def profile_network_forward(networks, num_iterations=50):
    """Profile the network forward pass."""
    print(f"Profiling network forward pass ({num_iterations} iterations)...")
    
    network = networks[0]
    inputs = [0.5, 0.3, 0.7, 0.1, 0.9]  # Match input_size
    
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(num_iterations):
        outputs = network.forward(inputs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_resource_interactions(environments, num_iterations=100):
    """Profile resource interaction calculations."""
    print(f"Profiling resource interactions ({num_iterations} iterations)...")
    
    env = environments[0]
    
    # Add more resources to stress test
    for i in range(20):
        x, y, z = (i % 10) * 0.1, (i // 10) * 0.1, 0.5
        env.add_nutrient_cluster((x, y, z), 0.15, ResourceType.CARBON, 0.5)
    
    position = (0.5, 0.5, 0.5)
    
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(num_iterations):
        resources = env.get_resources_in_range(position, 0.3)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_all():
    """Run all profiling tests."""
    # Create output directory
    os.makedirs("profile_results", exist_ok=True)
    
    # Redirect output to file
    original_stdout = sys.stdout
    with open("profile_results/performance_profile.txt", "w") as f:
        sys.stdout = f
        
        print("=" * 80)
        print("MYCELIUM NETWORK PERFORMANCE PROFILE")
        print("=" * 80)
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Profile environment creation
        environments = profile_environment_creation()
        print("\n" + "=" * 80 + "\n")
        
        # Profile environment update
        profile_environment_update(environments)
        print("\n" + "=" * 80 + "\n")
        
        # Profile network creation
        networks = profile_network_creation(environments)
        print("\n" + "=" * 80 + "\n")
        
        # Profile network forward pass
        profile_network_forward(networks)
        print("\n" + "=" * 80 + "\n")
        
        # Profile resource interactions
        profile_resource_interactions(environments)
    
    # Restore stdout
    sys.stdout = original_stdout
    print(f"Profiling complete. Results saved to profile_results/performance_profile.txt")


if __name__ == "__main__":
    profile_all()
