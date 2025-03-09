# Mycelium Network

A biomimetic neural network implementation inspired by the growth patterns, resource allocation, and adaptive behaviors of fungal mycelium networks.

## Overview

The Mycelium Network provides an innovative approach to neural networks that mimics the decentralized, adaptive properties of fungal mycelium. Unlike traditional neural networks with fixed architectures, mycelium networks can grow, adapt, and reshape themselves based on environmental factors and the tasks they're solving.

Key features include:
- **Dynamic growth**: Networks can add new nodes and connections during operation
- **Resource allocation**: Nodes distribute resources based on their utility
- **Chemical signaling**: Communication between nodes via signal propagation
- **Environmental awareness**: Networks operate within simulated environments that can contain resources and obstacles

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mycelium_network.git
cd mycelium_network

# Setup a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
mycelium_network/
├── examples/                  # Example scripts and utilities
│   ├── reorganized/           # Organized examples by category
│   │   ├── advanced/          # Advanced utilization examples
│   │   ├── basic/             # Basic task examples
│   │   ├── utils/             # Utility functions
│   │   └── visualization/     # Visualization examples
├── mycelium/                  # Core package
│   ├── __init__.py            # Package initialization
│   ├── environment.py         # Environment implementation
│   ├── network.py             # Network implementation
│   ├── node.py                # Node implementation
│   └── tasks/                 # Task-specific implementations
│       ├── __init__.py
│       ├── anomaly_detector.py
│       ├── classifier.py
│       └── regressor.py
├── requirements.txt           # Project dependencies
├── README.md                  # This documentation
└── venv/                      # Virtual environment (after setup)
```

## Quick Start

Follow these steps to get started with Mycelium Network:

1. **Setup the environment** as described in the Installation section.

2. **Run one of the example scripts** to see the network in action:

```bash
# Run a basic classification example using the Iris dataset
python examples/reorganized/basic/classification_example.py

# Run a regression example using synthetic data
python examples/reorganized/basic/regression_example.py

# Run an anomaly detection example
python examples/reorganized/basic/anomaly_detection_example.py

# Run an advanced example with environmental adaptation
python examples/reorganized/advanced/environment_adaptation_example.py
```

3. **Create your own implementation** based on the examples:

```python
from mycelium import MyceliumClassifier, Environment

# Create an environment
env = Environment(dimensions=4)

# Initialize and train a classifier
classifier = MyceliumClassifier(
    input_size=4,
    num_classes=2,
    hidden_nodes=20,
    environment=env
)

# Train the classifier
history = classifier.fit(X_train, y_train, epochs=15)

# Make predictions
predictions = classifier.predict(X_test)
```

## Components

### Environment

The `Environment` class provides the spatial context for the mycelium network:

```python
from mycelium import Environment

# Create an environment with 2D space
env = Environment(dimensions=2)

# Add resources and obstacles
env.add_resource((0.3, 0.7), 1.5)
env.add_obstacle((0.5, 0.5), 0.1)

# Create a grid of resources for testing
env.create_grid_resources(grid_size=5, resource_value=1.0)
```

### Network

The `AdvancedMyceliumNetwork` is the core component that implements the network functionality:

```python
from mycelium import AdvancedMyceliumNetwork, Environment

# Create an environment
env = Environment(dimensions=3)

# Create a network
network = AdvancedMyceliumNetwork(
    environment=env,
    input_size=3,
    output_size=1,
    initial_nodes=20
)

# Process inputs through the network
outputs = network.forward([0.5, 0.7, 0.3])

# Get network statistics
stats = network.get_network_statistics()
```

### Task-Specific Interfaces

For specific machine learning tasks, use these specialized classes:

```python
# Classification
from mycelium import MyceliumClassifier
classifier = MyceliumClassifier(input_size=4, num_classes=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Regression
from mycelium import MyceliumRegressor
regressor = MyceliumRegressor(input_size=2, output_size=1)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Anomaly Detection
from mycelium import MyceliumAnomalyDetector
detector = MyceliumAnomalyDetector(input_size=10, contamination=0.1)
detector.fit(X_train)
anomalies = detector.predict(X_test)
```

## Visualization

The examples include utilities for visualizing networks and their performance:

```python
# Import visualization utilities
from examples.reorganized.utils.visualization import (
    save_network_visualization_data,
    generate_html_visualization,
    plot_training_history
)

# Save network data for visualization
save_network_visualization_data(network, "network_data.json")

# Generate interactive HTML visualization
generate_html_visualization("network_data.json", "network_visualization.html")

# Plot training history
plot_training_history(classifier.train_history, "training_history.png")
```

## Advanced Features

### Environmental Adaptation

Networks can adapt to changing environmental conditions:

```python
# Create an enhanced environment with changing conditions
from examples.reorganized.advanced.environment_adaptation_example import EnhancedEnvironment

env = EnhancedEnvironment(dimensions=2)
env.moisture_level = 0.7
env.temperature = 0.5
env.add_nutrient_cluster((0.3, 0.3), 0.15, "carbon", 1.2)

# Create an adaptive network
network = AdaptiveMyceliumNetwork(environment=env)

# Run simulation with adaptation
for i in range(30):
    inputs = [random.random() for _ in range(2)]
    outputs = network.forward(inputs)
    print(f"Growth rate: {network.growth_rate}")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'mycelium'**
   - Make sure you're running the examples from the main directory
   - Check that the virtual environment is activated

2. **ValueError: Expected {input_size} inputs, got {len(inputs)}**
   - Ensure your input vectors match the dimensionality of the network

3. **Memory issues with large networks**
   - Reduce the number of initial nodes or growth rate
   - Consider running on a machine with more memory

### Performance Tips

1. **Faster training:**
   - Use fewer initial nodes and let the network grow adaptively
   - Set higher learning rates initially, then decrease over time

2. **Better accuracy:**
   - Create environments with meaningful resource distribution
   - Use obstacles to shape the network architecture
   - Increase training epochs to allow for more adaptation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
