# Mycelium Network

A biomimetic neural network implementation inspired by the growth patterns, resource allocation, and adaptive behaviors of fungal mycelium networks.

## Overview

The Mycelium Network provides an innovative approach to neural networks that mimics the decentralized, adaptive properties of fungal mycelium. Unlike traditional neural networks with fixed architectures, mycelium networks can grow, adapt, and reshape themselves based on environmental factors and the tasks they're solving.

Key features include:
- **Dynamic growth**: Networks can add new nodes and connections during operation
- **Resource allocation**: Nodes distribute resources based on their utility
- **Chemical signaling**: Communication between nodes via signal propagation
- **Environmental awareness**: Networks operate within simulated environments that can contain resources and obstacles
- **Enhanced ecosystem**: Complex multi-organism environment with plants, herbivores, and decomposers
- **Machine learning integration**: Reinforcement learning and transfer learning capabilities
- **Adaptive specialization**: Nodes can specialize for different functions based on environmental conditions

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
│   ├── enhanced_demo.py       # Demo for enhanced environment
│   ├── ecosystem_demo.py      # Demo for ecosystem simulation
│   ├── ml_integration_demo.py # Demo for machine learning capabilities
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
│   ├── enhanced/              # Enhanced components
│   │   ├── __init__.py
│   │   ├── rich_environment.py # Advanced environment implementation
│   │   ├── adaptive_network.py # Adaptive network implementation
│   │   ├── resource.py        # Enhanced resource types
│   │   ├── ecosystem/         # Ecosystem simulation
│   │   │   ├── __init__.py
│   │   │   ├── ecosystem.py   # Full ecosystem implementation
│   │   │   ├── interaction.py # Organism interaction registry
│   │   │   ├── organisms/     # Individual organism types
│   │   │   │   ├── base.py    # Base organism class
│   │   │   │   ├── plant.py   # Plant organisms
│   │   │   │   ├── herbivore.py # Herbivore organisms
│   │   │   │   └── decomposer.py # Decomposer organisms
│   │   ├── ml/               # Machine learning components
│   │   │   ├── __init__.py
│   │   │   ├── reinforcement.py # Reinforcement learning
│   │   │   └── transfer.py    # Transfer learning
│   └── tasks/                 # Task-specific implementations
│       ├── __init__.py
│       ├── anomaly_detector.py
│       ├── classifier.py
│       └── regressor.py
├── tools/                     # Utility tools
│   └── profile_performance.py # Performance profiling tool
├── visualizations/            # Generated visualizations from demos
│   ├── ecosystem_simulation.png  # Ecosystem demo visualization
│   └── rl_training_results.png   # ML demo visualization
├── requirements.txt           # Project dependencies
├── README.md                  # This documentation
└── venv/                      # Virtual environment (after setup)
```

## Quick Start

Follow these steps to get started with Mycelium Network:

1. **Setup the environment** as described in the Installation section.

2. **Run one of the example scripts** to see the network in action:

```bash
# Make sure to activate the virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the enhanced environment demo
python examples/enhanced_demo.py

# Run the ecosystem simulation demo
python examples/ecosystem_demo.py
# This will save a visualization plot to: visualizations/ecosystem_simulation.png

# Run the machine learning integration demo
python examples/ml_integration_demo.py
# This will save a visualization plot to: visualizations/rl_training_results.png

# Run a basic classification example using the Iris dataset
python examples/reorganized/basic/classification_example.py
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

### Basic Environment

The `Environment` class provides the spatial context for the mycelium network:

```python
from mycelium import Environment

# Create an environment with 2D space
env = Environment(dimensions=2)

# Add resources and obstacles
env.add_resource((0.3, 0.7), 1.5)
env.add_obstacle((0.5, 0.5), 0.1)
```

### Rich Environment

The enhanced `RichEnvironment` provides a more sophisticated environment model:

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType

# Create a 3D environment with terrain layers
env = RichEnvironment(dimensions=3, size=1.0, name="Demo Environment")

# Add diverse resource types
env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
env.add_nutrient_cluster((0.3, 0.3, 0.65), 0.15, ResourceType.WATER, 1.5)

# Create seasonal cycle
env.create_seasonal_cycle(year_length=24.0, intensity=0.7)

# Update environment over time
env.update(delta_time=0.5)
```

### Ecosystem Simulation

The ecosystem module provides a complex simulation of interacting organisms:

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.ecosystem.ecosystem import Ecosystem

# Create a rich environment
environment = RichEnvironment(dimensions=3, size=1.0)

# Create an ecosystem
ecosystem = Ecosystem(environment)

# Populate with organisms
ecosystem.populate_randomly(
    num_plants=15,
    num_herbivores=6,
    num_decomposers=4
)

# Run simulation
for i in range(30):
    stats = ecosystem.update(delta_time=0.5)
    print(f"Population: {stats['population']['total']} organisms")
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
```

### Adaptive Network

The enhanced `AdaptiveMyceliumNetwork` adapts to environmental conditions:

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create a rich environment
env = RichEnvironment(dimensions=3)

# Create an adaptive network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=10
)

# Process inputs and adapt to environment
for i in range(10):
    outputs = network.forward([0.7, 0.3])
    
    # Check adaptation levels
    stats = network.get_specialization_statistics()
    print(f"Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
```

### Machine Learning Integration

The ML components provide reinforcement learning and transfer learning capabilities:

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.reinforcement import ReinforcementLearner
from mycelium.enhanced.ml.transfer import TransferNetwork

# Create networks and environments
env1 = RichEnvironment()
env2 = RichEnvironment()
source_network = AdaptiveMyceliumNetwork(environment=env1)
target_network = AdaptiveMyceliumNetwork(environment=env2)

# Transfer knowledge between networks
transfer = TransferNetwork(similarity_threshold=0.5)
result = transfer.transfer_knowledge(source_network, target_network)
print(f"Knowledge transfer success: {result['success']}")
```

## Performance Optimization

The project includes a performance profiling tool to identify bottlenecks:

```bash
# Run the performance profiling tool
python tools/profile_performance.py
```

## Advanced Features

### Environmental Adaptation

Networks can adapt to changing environmental conditions:

```python
# Create environment with changing conditions
env = RichEnvironment(dimensions=2)

# Create an adaptive network
network = AdaptiveMyceliumNetwork(environment=env)

# Simulate changing conditions
env.factors.temperature = 0.8  # Hot environment
env.factors.moisture = 0.2     # Dry environment
env.update(delta_time=0.5)

# Run network with adaptation
for i in range(10):
    outputs = network.forward([0.5, 0.5])
    print(f"Temperature adaptation: {network.temperature_adaptation:.2f}")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'mycelium'**
   - Make sure you're running the examples from the main directory
   - Check that the virtual environment is activated

2. **ValueError: Expected {input_size} inputs, got {len(inputs)}**
   - Ensure your input vectors match the dimensionality of the network

3. **Memory issues with large ecosystems**
   - Reduce the number of organisms in the simulation
   - Consider running on a machine with more memory

4. **Permission denied when saving plots**
   - Visualization plots are saved to the 'visualizations/' directory in the repository
   - Make sure your user has write permissions to this directory
   - If issues persist, run the script with appropriate permissions or modify the save path in the demo scripts

## Future Roadmap

We're actively developing the following features:

1. **Performance Optimizations**
   - Implement spatial indexing for faster resource lookups
   - Optimize node connection algorithms
   - Add parallel processing for network updates

2. **Extended Ecosystem**
   - Add more organism types (e.g., carnivores, symbiotic organisms)
   - Implement more complex food webs and energy flows
   - Create visualization tools for ecosystem monitoring

3. **Advanced Learning Capabilities**
   - Implement deep reinforcement learning integration
   - Add evolutionary algorithms for network optimization
   - Develop species-level adaptation using genetic algorithms

4. **Visualization and Monitoring**
   - Create interactive 3D visualizations for networks and ecosystems
   - Implement real-time monitoring dashboards
   - Add export tools for analysis in other software

5. **Application Development**
   - Create specialized adapters for common AI tasks
   - Implement model comparison tools
   - Develop example applications for real-world problems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
