"""
Transfer learning implementation for the mycelium network.

This module provides the ability to transfer knowledge between 
different mycelium networks, allowing networks to learn from each other.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork


class TransferNetwork:
    """
    Transfer learning functionality for mycelium networks.
    
    This class enables knowledge transfer between networks, allowing
    a target network to benefit from the experience of a source network.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize the transfer network.
        
        Args:
            similarity_threshold: Minimum similarity for knowledge transfer
        """
        self.similarity_threshold = similarity_threshold
        
        # Transfer statistics
        self.transfer_stats = {
            "transfers": 0,
            "success_rate": 0,
            "average_similarity": 0,
            "total_knowledge_gain": 0
        }
    
    def calculate_similarity(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork
    ) -> float:
        """
        Calculate similarity between two networks.
        
        Args:
            source_network: Source network
            target_network: Target network
            
        Returns:
            Similarity score (0-1)
        """
        # Compare network structures
        struct_similarity = self._calculate_structural_similarity(
            source_network, target_network)
        
        # Compare specialization distributions
        spec_similarity = self._calculate_specialization_similarity(
            source_network, target_network)
        
        # Compare adaptation parameters
        adapt_similarity = self._calculate_adaptation_similarity(
            source_network, target_network)
        
        # Weight the similarities (structure is most important)
        return (
            struct_similarity * 0.5 +
            spec_similarity * 0.3 +
            adapt_similarity * 0.2
        )
    
    def _calculate_structural_similarity(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork
    ) -> float:
        """Calculate structural similarity between networks."""
        # Compare node counts
        source_nodes = len(source_network.nodes)
        target_nodes = len(target_network.nodes)
        
        # Size ratio (smaller to larger)
        size_ratio = min(source_nodes, target_nodes) / max(source_nodes, target_nodes)
        
        # Compare connectivity patterns
        source_connectivity = self._calculate_connectivity_density(source_network)
        target_connectivity = self._calculate_connectivity_density(target_network)
        
        # Connectivity similarity (1 - normalized difference)
        connectivity_similarity = 1 - min(1, abs(source_connectivity - target_connectivity) / max(source_connectivity, target_connectivity))
        
        # Input/output structure similarity
        io_similarity = self._compare_io_structure(source_network, target_network)
        
        # Combine metrics
        return (
            size_ratio * 0.4 +
            connectivity_similarity * 0.4 +
            io_similarity * 0.2
        )
    
    def _calculate_connectivity_density(self, network: AdaptiveMyceliumNetwork) -> float:
        """Calculate the connectivity density of a network."""
        total_connections = 0
        possible_connections = 0
        
        for node_id, node in network.nodes.items():
            total_connections += len(node.connections)
            possible_connections += len(network.nodes) - 1  # All except self
        
        if possible_connections == 0:
            return 0
        
        return total_connections / possible_connections
    
    def _compare_io_structure(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork
    ) -> float:
        """Compare input/output structure similarity."""
        # Compare input/output sizes
        input_ratio = min(len(source_network.input_nodes), len(target_network.input_nodes)) / \
                     max(len(source_network.input_nodes), len(target_network.input_nodes))
        
        output_ratio = min(len(source_network.output_nodes), len(target_network.output_nodes)) / \
                      max(len(source_network.output_nodes), len(target_network.output_nodes))
        
        return (input_ratio + output_ratio) / 2
    
    def _calculate_specialization_similarity(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork
    ) -> float:
        """Calculate similarity in node specialization distributions."""
        # Get specialized node counts
        source_specs = {}
        target_specs = {}
        
        for spec_type, nodes in source_network.specializations.items():
            source_specs[spec_type] = len(nodes)
        
        for spec_type, nodes in target_network.specializations.items():
            target_specs[spec_type] = len(nodes)
        
        # Calculate Jaccard similarity for specialization types
        common_types = set(source_specs.keys()) & set(target_specs.keys())
        all_types = set(source_specs.keys()) | set(target_specs.keys())
        
        type_similarity = len(common_types) / max(1, len(all_types))
        
        # Calculate distribution similarity for common types
        distribution_similarity = 0
        if common_types:
            similarities = []
            for spec_type in common_types:
                source_ratio = source_specs[spec_type] / max(1, len(source_network.nodes))
                target_ratio = target_specs[spec_type] / max(1, len(target_network.nodes))
                
                # Similarity as 1 - normalized difference
                ratio_similarity = 1 - min(1, abs(source_ratio - target_ratio) / max(source_ratio, target_ratio, 0.01))
                similarities.append(ratio_similarity)
            
            distribution_similarity = sum(similarities) / len(similarities)
        
        return type_similarity * 0.4 + distribution_similarity * 0.6
    
    def _calculate_adaptation_similarity(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork
    ) -> float:
        """Calculate similarity in adaptation parameters."""
        # Compare various adaptation parameters
        temp_similarity = 1 - abs(source_network.temperature_adaptation - target_network.temperature_adaptation)
        moisture_similarity = 1 - abs(source_network.moisture_adaptation - target_network.moisture_adaptation)
        drought_similarity = 1 - abs(source_network.drought_resistance - target_network.drought_resistance)
        stress_similarity = 1 - abs(source_network.stress_tolerance - target_network.stress_tolerance)
        
        return (temp_similarity + moisture_similarity + drought_similarity + stress_similarity) / 4
    
    def transfer_knowledge(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork,
        transfer_rate: float = 0.3
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from source network to target network.
        
        Args:
            source_network: Source network (experienced)
            target_network: Target network (learning)
            transfer_rate: Rate at which knowledge is transferred (0-1)
            
        Returns:
            Transfer statistics
        """
        # Calculate similarity
        similarity = self.calculate_similarity(source_network, target_network)
        
        # Track attempt
        self.transfer_stats["transfers"] += 1
        
        # Check similarity threshold
        if similarity < self.similarity_threshold:
            return {
                "success": False,
                "similarity": similarity,
                "reason": "Similarity below threshold"
            }
        
        # Calculate effective transfer rate (adjusted by similarity)
        effective_rate = transfer_rate * similarity
        
        # Transfer adaptation parameters
        knowledge_gain = self._transfer_adaptation_parameters(
            source_network, target_network, effective_rate)
        
        # Transfer specialized nodes
        specialization_gain = self._transfer_specializations(
            source_network, target_network, effective_rate)
        
        # Transfer resource efficiency
        efficiency_gain = self._transfer_resource_efficiency(
            source_network, target_network, effective_rate)
        
        # Track success
        self.transfer_stats["success_rate"] = ((self.transfer_stats["success_rate"] * 
                                              (self.transfer_stats["transfers"] - 1) + 1) / 
                                             self.transfer_stats["transfers"])
        
        self.transfer_stats["average_similarity"] = ((self.transfer_stats["average_similarity"] * 
                                                   (self.transfer_stats["transfers"] - 1) + similarity) / 
                                                  self.transfer_stats["transfers"])
        
        total_gain = knowledge_gain + specialization_gain + efficiency_gain
        self.transfer_stats["total_knowledge_gain"] += total_gain
        
        return {
            "success": True,
            "similarity": similarity,
            "knowledge_gain": total_gain,
            "effective_transfer_rate": effective_rate,
            "adaptation_gain": knowledge_gain,
            "specialization_gain": specialization_gain,
            "efficiency_gain": efficiency_gain
        }
    
    def _transfer_adaptation_parameters(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork,
        transfer_rate: float
    ) -> float:
        """Transfer adaptation parameters between networks."""
        # Calculate differences in adaptation parameters
        temp_diff = source_network.temperature_adaptation - target_network.temperature_adaptation
        moisture_diff = source_network.moisture_adaptation - target_network.moisture_adaptation
        drought_diff = source_network.drought_resistance - target_network.drought_resistance
        stress_diff = source_network.stress_tolerance - target_network.stress_tolerance
        
        # Apply transfer (partial movement toward source values)
        target_network.temperature_adaptation += temp_diff * transfer_rate
        target_network.moisture_adaptation += moisture_diff * transfer_rate
        target_network.drought_resistance += drought_diff * transfer_rate
        target_network.stress_tolerance += stress_diff * transfer_rate
        
        # Calculate total knowledge gain from adaptation transfer
        knowledge_gain = (abs(temp_diff) + abs(moisture_diff) + 
                         abs(drought_diff) + abs(stress_diff)) * transfer_rate / 4
        
        return knowledge_gain
    
    def _transfer_specializations(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork,
        transfer_rate: float
    ) -> float:
        """Transfer node specializations between networks."""
        # Determine specialization transfer amount
        knowledge_gain = 0
        
        for spec_type, source_nodes in source_network.specializations.items():
            # Calculate source specialization ratio
            source_ratio = len(source_nodes) / max(1, len(source_network.nodes))
            
            # Get target specialization ratio
            target_nodes = target_network.specializations.get(spec_type, [])
            target_ratio = len(target_nodes) / max(1, len(target_network.nodes))
            
            # Only transfer if source has more specialization
            if source_ratio > target_ratio:
                # Calculate how many more nodes to specialize
                ratio_diff = source_ratio - target_ratio
                target_conversion_count = int(ratio_diff * len(target_network.nodes) * transfer_rate)
                
                # Convert some regular nodes to this specialization
                converted = 0
                for node_id in list(target_network.regular_nodes):
                    if converted >= target_conversion_count:
                        break
                    
                    # Skip if already in a specialization
                    skip = False
                    for other_spec, other_nodes in target_network.specializations.items():
                        if node_id in other_nodes:
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    # Convert node
                    node = target_network.nodes[node_id]
                    
                    # Remove from regular nodes
                    target_network.regular_nodes.remove(node_id)
                    
                    # Add to specialization collection
                    if spec_type not in target_network.specializations:
                        target_network.specializations[spec_type] = []
                    
                    target_network.specializations[spec_type].append(node_id)
                    
                    # Apply specialization-specific changes
                    if spec_type == 'storage':
                        node.resource_level *= 1.5
                        node.longevity += 10
                    elif spec_type == 'processing':
                        node.sensitivity *= 1.2
                        node.adaptability *= 1.3
                    elif spec_type == 'sensor':
                        node.sensitivity *= 1.5
                        node.energy *= 0.8
                    
                    converted += 1
                
                # Add to knowledge gain
                knowledge_gain += converted / max(1, len(target_network.nodes))
        
        return knowledge_gain
    
    def _transfer_resource_efficiency(
        self,
        source_network: AdaptiveMyceliumNetwork,
        target_network: AdaptiveMyceliumNetwork,
        transfer_rate: float
    ) -> float:
        """Transfer resource efficiency knowledge."""
        knowledge_gain = 0
        
        for resource_type, source_efficiency in source_network.resource_efficiency.items():
            if resource_type in target_network.resource_efficiency:
                target_efficiency = target_network.resource_efficiency[resource_type]
                
                # Only transfer if source is more efficient
                if source_efficiency > target_efficiency:
                    efficiency_diff = source_efficiency - target_efficiency
                    
                    # Apply transfer
                    target_network.resource_efficiency[resource_type] += efficiency_diff * transfer_rate
                    
                    # Add to knowledge gain
                    knowledge_gain += efficiency_diff * transfer_rate / len(source_network.resource_efficiency)
        
        return knowledge_gain
