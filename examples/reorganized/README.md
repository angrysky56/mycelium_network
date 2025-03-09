# Mycelium Network Examples

This directory contains examples demonstrating the capabilities of the Mycelium Network framework.

## Directory Structure

- `basic/` - Basic examples of the core functionality
  - `classification_example.py` - Classification using mycelium networks
  - `regression_example.py` - Regression using mycelium networks
  - `anomaly_detection_example.py` - Anomaly detection using mycelium networks

- `advanced/` - Advanced examples with enhanced functionality
  - `environment_adaptation_example.py` - Demonstrates network adaptation to environmental changes

- `utils/` - Utility functions for data handling and visualization
  - `data_utils.py` - Data loading, preprocessing, and generation utilities
  - `visualization/` - Network visualization and performance plotting utilities

## Running Examples

You can run any example from the main repository directory:

```bash
# Make sure you're in the main repository directory
cd /path/to/mycelium_network

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run a basic example
python examples/reorganized/basic/classification_example.py

# Run an advanced example
python examples/reorganized/advanced/environment_adaptation_example.py
```

## Example Outputs

The examples will generate console output showing performance metrics, network statistics, and other relevant information. Some examples will also generate visualization files in the current directory that can be opened in a web browser.

## Creating Your Own Examples

To create your own examples, use the existing examples as templates. The key components you'll need are:

1. Import the necessary modules from the mycelium package
2. Create an environment (optional but recommended)
3. Create and configure a network (or task-specific class like MyceliumClassifier)
4. Train and evaluate the network
5. Visualize the results (optional)

## Contributing Examples

If you create an interesting example, consider contributing it to the repository:

1. Make sure your example is well-documented with comments
2. Include sample output in comments or a separate README
3. Place it in the appropriate directory based on complexity
4. Submit a pull request with a description of what your example demonstrates
