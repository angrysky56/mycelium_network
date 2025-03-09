#!/usr/bin/env python3
"""
Script to inspect and debug the training method of the Mycelium Network.

This script prints out relevant portions of the train method to help
diagnose why the training process isn't working as expected.
"""

import os
import sys
import inspect
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.network import AdvancedMyceliumNetwork


def print_train_method():
    """Print the train method code for inspection."""
    print("Inspecting the train method of AdvancedMyceliumNetwork:")
    print("====================================================")
    
    # Get the source code of the train method
    train_method = inspect.getsource(AdvancedMyceliumNetwork.train)
    
    print(train_method)
    print("\nInspecting other related methods:")
    print("================================")
    
    # Check other methods that might be involved in training
    for method_name in ["forward", "_process_signals", "_update_network_state", 
                        "_allocate_resources", "_grow_network", "_prune_network", 
                        "_adapt_connections", "_get_incoming_connections"]:
        try:
            method = getattr(AdvancedMyceliumNetwork, method_name)
            method_source = inspect.getsource(method)
            print(f"\n{method_name} method:")
            print("-" * (len(method_name) + 8))
            print(method_source)
        except (AttributeError, TypeError):
            print(f"Could not get source for {method_name}")


def print_node_process_signal():
    """Print the process_signal method of MyceliumNode."""
    from mycelium.node import MyceliumNode
    
    print("\nInspecting MyceliumNode's process_signal method:")
    print("===============================================")
    
    try:
        method = MyceliumNode.process_signal
        method_source = inspect.getsource(method)
        print(method_source)
    except (AttributeError, TypeError):
        print("Could not get source for process_signal")


def main():
    """Run the inspection script."""
    print_train_method()
    print_node_process_signal()


if __name__ == "__main__":
    main()
