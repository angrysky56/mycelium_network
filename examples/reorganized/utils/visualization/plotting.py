#!/usr/bin/env python3
"""
Data Plotting Utilities for Mycelium Network

This module provides functions for creating plots and visualizations
of network performance, training history, and decision boundaries.
"""

from typing import List, Tuple, Dict, Any, Optional


def plot_training_history(history: Dict[str, List[float]], filename: str = None) -> None:
    """
    Generate a plot of training history metrics.
    
    Parameters:
    -----------
    history : Dict[str, List[float]]
        Dictionary of training history metrics
    filename : str, optional
        Path to save the plot (if None, display the plot)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(nrows=len(history), figsize=(10, 3 * len(history)))
        
        # Convert to list if only one metric
        if len(history) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(history.items()):
            ax = axes[i]
            ax.plot(values)
            ax.set_title(f'{metric_name.capitalize()} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.capitalize())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_decision_boundary(
    classifier,
    X: List[List[float]],
    y: List[int],
    feature_indices: Tuple[int, int] = (0, 1),
    resolution: int = 100,
    filename: str = None
) -> None:
    """
    Plot the decision boundary for a classifier.
    
    Parameters:
    -----------
    classifier : MyceliumClassifier
        Trained classifier
    X : List[List[float]]
        Feature vectors
    y : List[int]
        Class labels
    feature_indices : Tuple[int, int]
        Indices of features to use for the plot (2D only)
    resolution : int
        Resolution of the decision boundary grid
    filename : str, optional
        Path to save the plot (if None, display the plot)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract the two features to plot
        idx1, idx2 = feature_indices
        X_plot = [[sample[idx1], sample[idx2]] for sample in X]
        
        # Determine plot boundaries
        x_min, x_max = min(x[0] for x in X_plot), max(x[0] for x in X_plot)
        y_min, y_max = min(x[1] for x in X_plot), max(x[1] for x in X_plot)
        
        # Add margin
        margin = 0.1
        x_min -= margin * (x_max - x_min)
        x_max += margin * (x_max - x_min)
        y_min -= margin * (y_max - y_min)
        y_max += margin * (y_max - y_min)
        
        # Create a meshgrid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create feature vectors for all grid points
        grid_points = []
        for i in range(resolution):
            for j in range(resolution):
                # Create a full feature vector with zeros for non-plotted dimensions
                feature_vector = [0.0] * len(X[0])
                feature_vector[idx1] = xx[i, j]
                feature_vector[idx2] = yy[i, j]
                grid_points.append(feature_vector)
        
        # Get predictions for grid points
        Z = classifier.predict(grid_points)
        
        # Reshape Z to match the meshgrid shape
        Z = np.array(Z).reshape(resolution, resolution)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Plot the data points
        unique_classes = set(y)
        for cls in unique_classes:
            indices = [i for i, label in enumerate(y) if label == cls]
            plt.scatter(
                [X[i][idx1] for i in indices],
                [X[i][idx2] for i in indices],
                alpha=0.8,
                label=f'Class {cls}'
            )
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(f'Feature {idx1}')
        plt.ylabel(f'Feature {idx2}')
        plt.title('Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib and NumPy are required for plotting. Install with: pip install matplotlib numpy")


def plot_network_performance_comparison(
    performance_data: List[Dict],
    metric_name: str,
    title: str = "Performance Comparison",
    filename: str = None
) -> None:
    """
    Plot performance comparison between different network configurations.
    
    Parameters:
    -----------
    performance_data : List[Dict]
        List of dictionaries with 'name' and 'values' keys
    metric_name : str
        Name of the metric being compared
    title : str
        Plot title
    filename : str, optional
        Path to save the plot (if None, display the plot)
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot each configuration
        for i, data in enumerate(performance_data):
            plt.plot(
                range(1, len(data['values']) + 1),
                data['values'],
                label=data['name'],
                marker='o' if len(data['values']) < 15 else None
            )
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_resource_distribution(
    network,
    title: str = "Resource Distribution in Mycelium Network",
    filename: str = None
) -> None:
    """
    Plot resource distribution across nodes in the network.
    
    Parameters:
    -----------
    network : AdvancedMyceliumNetwork
        The mycelium network to visualize
    title : str
        Plot title
    filename : str, optional
        Path to save the plot (if None, display the plot)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect resource levels by node type
        resources = {
            'input': [],
            'output': [],
            'regular': []
        }
        
        for node_id, node in network.nodes.items():
            node_type = 'input' if node_id in network.input_nodes else 'output' if node_id in network.output_nodes else 'regular'
            resources[node_type].append(node.resource_level)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        plt.subplot(1, 2, 1)
        
        # Bar plot of average resources by node type
        avg_resources = {k: np.mean(v) if v else 0 for k, v in resources.items()}
        plt.bar(avg_resources.keys(), avg_resources.values(), color=['#6baed6', '#fd8d3c', '#74c476'])
        plt.title('Average Resource Level by Node Type')
        plt.ylabel('Resource Level')
        plt.grid(True, alpha=0.3)
        
        # Histogram of resource distribution
        plt.subplot(1, 2, 2)
        all_resources = []
        for r in resources.values():
            all_resources.extend(r)
            
        plt.hist(all_resources, bins=15, alpha=0.7, color='#9467bd')
        plt.title('Resource Distribution Across All Nodes')
        plt.xlabel('Resource Level')
        plt.ylabel('Number of Nodes')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib and NumPy are required for plotting. Install with: pip install matplotlib numpy")
