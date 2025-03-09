"""
Performance visualization tools for mycelium networks.

This module provides tools for visualizing mycelium network performance,
including training metrics, genetic optimization progress, and more.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


class PerformanceVisualizer:
    """
    Visualizer for network performance and optimization results.
    
    This class provides methods for creating visualizations of
    network performance, genetic algorithm progress, etc.
    """
    
    def __init__(self, output_dir='./visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization state
        self.fig = None
        self.ax = None
    
    def plot_training_history(self, history, title=None, save_path=None, show=True):
        """
        Plot training history (error over epochs).
        
        Args:
            history: List of error values per epoch
            title: Optional title for the visualization
            save_path: Path to save the visualization (or None to not save)
            show: Whether to display the visualization
            
        Returns:
            Figure and axes objects
        """
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot error over epochs
        epochs = list(range(1, len(history) + 1))
        self.ax.plot(epochs, history, 'b-', linewidth=2)
        
        # Set labels and title
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Error')
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title('Training History')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save if requested
        if save_path:
            save_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return self.fig, self.ax
    
    def plot_genetic_optimization(self, fitness_history, title=None, save_path=None, show=True):
        """
        Plot genetic algorithm optimization progress.
        
        Args:
            fitness_history: List of dictionaries with 'generation', 'average', and 'best' keys
            title: Optional title for the visualization
            save_path: Path to save the visualization (or None to not save)
            show: Whether to display the visualization
            
        Returns:
            Figure and axes objects
        """
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        generations = [entry['generation'] for entry in fitness_history]
        avg_fitness = [entry['average'] for entry in fitness_history]
        best_fitness = [entry['best'] for entry in fitness_history]
        
        # Plot average and best fitness
        self.ax.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        self.ax.plot(generations, best_fitness, 'r-', linewidth=2, label='Best Fitness')
        
        # Set labels and title
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title('Genetic Algorithm Optimization')
        
        # Add legend and grid
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save if requested
        if save_path:
            save_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return self.fig, self.ax
    
    def compare_performance(self, results, labels, title=None, save_path=None, show=True):
        """
        Compare performance of different network implementations.
        
        Args:
            results: Dictionary of {metric_name: [values_for_each_label]}
            labels: List of labels for each implementation
            title: Optional title for the visualization
            save_path: Path to save the visualization (or None to not save)
            show: Whether to display the visualization
            
        Returns:
            Figure and axes objects
        """
        # Create figure with subplots for each metric
        num_metrics = len(results)
        self.fig, self.axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        
        # Ensure axes is a list even with one metric
        if num_metrics == 1:
            self.axes = [self.axes]
        
        # Set overall title
        if title:
            self.fig.suptitle(title, fontsize=16)
        else:
            self.fig.suptitle('Performance Comparison', fontsize=16)
        
        # Add padding between subplots
        self.fig.tight_layout(pad=3.0)
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(results.items()):
            ax = self.axes[i]
            
            # Bar plot for this metric
            x = np.arange(len(labels))
            ax.bar(x, values, width=0.6, alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Implementation')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            
            # Set x-tick labels
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Add values on top of bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.05 * max(values), f'{v:.2f}', 
                      ha='center', va='bottom', fontweight='bold')
        
        # Save if requested
        if save_path:
            save_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return self.fig, self.axes
    
    def plot_adaptation_history(self, adaptation_history, metrics=None, title=None, 
                              save_path=None, show=True):
        """
        Plot network adaptation over time.
        
        Args:
            adaptation_history: List of adaptation state dictionaries
            metrics: List of metrics to plot (or None for default)
            title: Optional title for the visualization
            save_path: Path to save the visualization (or None to not save)
            show: Whether to display the visualization
            
        Returns:
            Figure and axes objects
        """
        # Default metrics to plot
        if metrics is None:
            metrics = ['temperature_adaptation', 'moisture_adaptation', 'drought_resistance']
        
        # Filter metrics that exist in the history
        available_metrics = []
        for metric in metrics:
            if metric in adaptation_history[0]:
                available_metrics.append(metric)
        
        if not available_metrics:
            # Try looking in nested dictionaries
            for metric in metrics:
                for key in adaptation_history[0].keys():
                    if isinstance(adaptation_history[0][key], dict) and metric in adaptation_history[0][key]:
                        available_metrics.append(f"{key}.{metric}")
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot each metric
        iterations = [entry['iteration'] for entry in adaptation_history]
        
        for metric in available_metrics:
            # Handle nested metrics
            if '.' in metric:
                parent, child = metric.split('.')
                values = [entry[parent][child] for entry in adaptation_history]
                label = f"{parent.capitalize()} {child.replace('_', ' ').capitalize()}"
            else:
                values = [entry[metric] for entry in adaptation_history]
                label = metric.replace('_', ' ').capitalize()
            
            self.ax.plot(iterations, values, linewidth=2, label=label)
        
        # Set labels and title
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Adaptation Value')
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title('Network Adaptation History')
        
        # Add legend and grid
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save if requested
        if save_path:
            save_path = os.path.join(self.output_dir, save_path) if not os.path.isabs(save_path) else save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()
        
        return self.fig, self.ax
