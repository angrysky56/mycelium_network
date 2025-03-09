#!/usr/bin/env python3
"""
Demonstration of the spatial indexing implementation for the mycelium network.

This script compares the performance of the standard environment vs.
the spatially-indexed environment for resource lookups.
"""

import os
import sys
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.environment import Environment
from mycelium.spatial.spatial_index import SpatialIndex


def create_environment_with_resources(num_resources=100, clustered=False):
    """Create an environment with randomly distributed resources."""
    env = Environment(dimensions=2)
    
    if clustered:
        # Create clustered resources
        num_clusters = 5
        resources_per_cluster = num_resources // num_clusters
        
        for _ in range(num_clusters):
            # Random cluster center
            center_x = random.random()
            center_y = random.random()
            
            # Add resources around center
            for _ in range(resources_per_cluster):
                # Random offset from center
                offset_x = random.gauss(0, 0.1)  # Gaussian distribution
                offset_y = random.gauss(0, 0.1)
                
                # Ensure position is within bounds
                x = max(0, min(1, center_x + offset_x))
                y = max(0, min(1, center_y + offset_y))
                
                # Random resource amount
                amount = random.uniform(0.5, 2.0)
                
                # Add resource
                env.add_resource((x, y), amount)
    else:
        # Create randomly distributed resources
        for _ in range(num_resources):
            x = random.random()
            y = random.random()
            amount = random.uniform(0.5, 2.0)
            env.add_resource((x, y), amount)
    
    return env


def benchmark_resource_lookup(env, num_queries=1000, radius=0.1):
    """Benchmark resource lookup performance."""
    # Generate random query points
    queries = [(random.random(), random.random()) for _ in range(num_queries)]
    
    # Time the standard lookup
    start_time = time.time()
    for query_point in queries:
        resources = env.get_resources_in_range(query_point, radius)
    elapsed_time = time.time() - start_time
    
    return elapsed_time


def benchmark_with_spatial_index(env, num_queries=1000, radius=0.1):
    """Benchmark resource lookup with spatial indexing."""
    # Create spatial index
    spatial_index = SpatialIndex(dimensions=2)
    
    # Add resources to spatial index
    for pos, amount in env.resources.items():
        item_id = f"resource_{hash(pos)}"
        spatial_index.insert(item_id, pos, amount)
    
    # Generate random query points
    queries = [(random.random(), random.random()) for _ in range(num_queries)]
    
    # Time the lookup using spatial index
    start_time = time.time()
    for query_point in queries:
        resources = spatial_index.query_range(query_point, radius)
    elapsed_time = time.time() - start_time
    
    return elapsed_time


def run_benchmark():
    """Run the benchmark comparison."""
    print("Spatial Indexing Performance Benchmark")
    print("======================================")
    
    # Parameters
    resource_counts = [100, 500, 1000, 5000]
    query_count = 1000
    search_radius = 0.1
    
    # Results
    standard_times = []
    indexed_times = []
    
    for resource_count in resource_counts:
        print(f"\nRunning with {resource_count} resources...")
        
        # Create environment
        env = create_environment_with_resources(resource_count, clustered=True)
        
        # Benchmark standard lookup
        standard_time = benchmark_resource_lookup(env, query_count, search_radius)
        standard_times.append(standard_time)
        print(f"  Standard lookup: {standard_time:.4f} seconds")
        
        # Benchmark with spatial index
        indexed_time = benchmark_with_spatial_index(env, query_count, search_radius)
        indexed_times.append(indexed_time)
        print(f"  Spatial index lookup: {indexed_time:.4f} seconds")
        
        # Calculate speedup
        speedup = standard_time / indexed_time if indexed_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
    
    # Visualize results
    output_dir = '/home/ty/Repositories/ai_workspace/mycelium_network/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'Standard Lookup Time (s)': standard_times,
        'Spatial Index Lookup Time (s)': indexed_times,
        'Speedup': [s/i for s, i in zip(standard_times, indexed_times)]
    }
    
    # Plot time comparison
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(resource_counts))
    
    plt.bar(x - bar_width/2, standard_times, bar_width, label='Standard Lookup')
    plt.bar(x + bar_width/2, indexed_times, bar_width, label='Spatial Index Lookup')
    
    plt.xlabel('Number of Resources')
    plt.ylabel('Time (seconds)')
    plt.title('Lookup Performance Comparison')
    plt.xticks(x, resource_counts)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_index_performance.png'), dpi=300, bbox_inches='tight')
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    speedups = [s/i for s, i in zip(standard_times, indexed_times)]
    
    plt.plot(resource_counts, speedups, 'r-o', linewidth=2)
    
    plt.xlabel('Number of Resources')
    plt.ylabel('Speedup Factor')
    plt.title('Spatial Indexing Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_index_speedup.png'), dpi=300, bbox_inches='tight')
    
    print("\nResults saved to visualizations directory.")


if __name__ == "__main__":
    run_benchmark()
