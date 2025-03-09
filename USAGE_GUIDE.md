# Mycelium Network Usage Guide

This guide provides examples and best practices for using the Mycelium Network project.

## Basic Concepts

The Mycelium Network project simulates fungal mycelium networks as a new type of neural network. Key components:

1. **Environment**: The space where networks operate
2. **Network**: The mycelium network that grows and adapts
3. **Nodes**: Individual cells in the network
4. **Resources**: Energy and nutrients for growth
5. **Signals**: Communication between nodes

## Quick Start Examples

### Create a Basic Environment and Network

```python
from mycelium.environment import Environment
from mycelium.network import AdvancedMyceliumNetwork

# Create environment
env = Environment(dimensions=2)

# Add some resources
env.add_resource((0.5, 0.5), 1.0)  # Add resource at position (0.5, 0.5)

# Create network
network = AdvancedMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=10
)

# Forward pass
inputs = [0.7, 0.3]
outputs = network.forward(inputs)
print(f"Outputs: {outputs}")
```

### Create a Rich Environment with Enhanced Features

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create rich environment
env = RichEnvironment(dimensions=3)

# Add different resource types
env.add_nutrient_cluster((0.5, 0.5, 0.6), 0.2, ResourceType.CARBON, 2.0)
env.add_nutrient_cluster((0.3, 0.3, 0.6), 0.15, ResourceType.WATER, 1.5)

# Create adaptive network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=10
)

# Forward pass
inputs = [0.7, 0.3]
outputs = network.forward(inputs)
print(f"Outputs: {outputs}")
```

### Use Batch Processing for Performance

```python
from mycelium.optimized.batch_network import BatchProcessingNetwork

# Create batch network
network = BatchProcessingNetwork(
    input_size=2,
    output_size=1,
    initial_nodes=10,
    batch_size=5  # Process signals in batches of 5
)

# Forward pass
inputs = [0.7, 0.3]
outputs = network.forward(inputs)
print(f"Outputs: {outputs}")
```

### Optimize Network with Genetic Algorithm

```python
from mycelium.enhanced.ml.genetic import GeneticOptimizer, NetworkGenome
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create network template
template = AdaptiveMyceliumNetwork(
    input_size=2,
    output_size=1,
    initial_nodes=5
)

# Create optimizer
optimizer = GeneticOptimizer(
    population_size=20,
    mutation_rate=0.2,
    crossover_rate=0.7
)

# Initialize population
optimizer.initialize_population(template)

# Define fitness function for XOR problem
def xor_fitness(network):
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]
    
    total_error = 0
    for inputs, target in zip(xor_inputs, xor_targets):
        output = network.forward(inputs)[0]
        error = (output - target) ** 2
        total_error += error
    
    return 1 / (1 + total_error)

# Run optimization for several generations
for generation in range(10):
    # Evaluate fitness
    avg_fitness = optimizer.evaluate_fitness(xor_fitness)
    
    # Get best network
    best_genome, best_fitness = optimizer.get_best()
    
    print(f"Generation {generation+1}: Avg fitness = {avg_fitness:.4f}, Best fitness = {best_fitness:.4f}")
    
    # Evolve population
    optimizer.evolve()

# Get best network
best_network = best_genome.to_network()
```

## Best Practices

1. **Always Use the Virtual Environment**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run From Project Root Directory**
   ```bash
   cd mycelium_network
   python examples/batch_processing_demo.py
   ```

3. **Match Input/Output Dimensions**
   ```python
   # If you define a network with 2 inputs
   network = AdaptiveMyceliumNetwork(input_size=2, output_size=1)
   
   # You must provide exactly 2 inputs
   outputs = network.forward([0.5, 0.7])  # Correct
   ```

4. **Import from Correct Paths**
   ```python
   # Correct imports
   from mycelium.enhanced.rich_environment import RichEnvironment  # Correct
   from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
   
   # Incorrect imports
   from mycelium.enhanced.environment import RichEnvironment  # Wrong
   ```

5. **Use Health Check Script for Troubleshooting**
   ```bash
   python tools/health_check.py
   ```

## Common Patterns

### Creating a Custom Task

```python
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

class MyceliumRegressor(AdaptiveMyceliumNetwork):
    """Custom regressor using mycelium network."""
    
    def __init__(self, input_size, hidden_nodes=10, environment=None):
        super().__init__(
            environment=environment,
            input_size=input_size,
            output_size=1,  # Regression has single output
            initial_nodes=hidden_nodes
        )
    
    def fit(self, X, y, epochs=10):
        """Train the regressor."""
        errors = []
        
        for epoch in range(epochs):
            epoch_error = 0
            
            # Train on each sample
            for inputs, target in zip(X, y):
                # Forward pass
                output = self.forward(inputs)[0]
                
                # Calculate error
                error = target - output
                epoch_error += error ** 2
                
                # Backpropagate error (simplified)
                self._adjust_weights(inputs, error)
            
            # Average error
            avg_error = epoch_error / len(X)
            errors.append(avg_error)
            
        return errors
    
    def _adjust_weights(self, inputs, error):
        """Adjust weights based on error."""
        # Implementation depends on specific learning algorithm
        pass
    
    def predict(self, X):
        """Make predictions for new data."""
        return [self.forward(inputs)[0] for inputs in X]
```

### Seasonal Simulation

```python
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork

# Create environment with seasons
env = RichEnvironment(dimensions=3)
env.create_seasonal_cycle(year_length=24.0, intensity=0.7)

# Create network
network = AdaptiveMyceliumNetwork(
    environment=env,
    input_size=2,
    output_size=1,
    initial_nodes=10
)

# Run simulation
for time_step in range(24):
    # Update environment
    env.update(delta_time=1.0)
    
    # Get current season
    year_phase = (env.time % env.year_length) / env.year_length
    season_idx = int(year_phase * 4) % 4
    season_names = ["Spring", "Summer", "Fall", "Winter"]
    current_season = season_names[season_idx]
    
    # Get temperature and moisture
    temperature = env.factors.temperature
    moisture = env.factors.moisture
    
    print(f"Time {time_step}: {current_season}, Temp: {temperature:.2f}, Moisture: {moisture:.2f}")
    
    # Run network with random inputs
    inputs = [0.5, 0.5]
    outputs = network.forward(inputs)
    
    # Check adaptation
    stats = network.get_specialization_statistics()
    print(f"  Temperature adaptation: {stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"  Moisture adaptation: {stats['adaptation']['moisture_adaptation']:.3f}")
```

## Debugging

If you encounter issues, try these steps:

1. **Run the health check script**:
   ```bash
   python tools/health_check.py
   ```

2. **Verify your environment**:
   ```bash
   python quick_test.py
   ```

3. **Check import paths**:
   ```python
   # Make sure you're using correct imports
   from mycelium.enhanced.rich_environment import RichEnvironment  # Correct
   from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
   ```

4. **Match input/output dimensions**:
   ```python
   network = AdaptiveMyceliumNetwork(input_size=2, output_size=1)
   outputs = network.forward([0.5, 0.7])  # Must be 2 inputs
   ```

5. **Run specific tests**:
   ```bash
   python -m pytest tests/test_batch_network.py
   python -m pytest tests/test_genetic.py
   python -m pytest tests/test_spatial_index.py
   ```

## Further Reading

For more information, see:
- Examples in the `examples/` directory
- The `README.md` file
- The `CONTRIBUTING.md` guide for development details

Happy experimenting with your mycelium networks!
