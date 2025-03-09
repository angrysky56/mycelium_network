"""
Interaction registry for the enhanced ecosystem.

This module provides tools for registering and managing interactions
between different organism types in the ecosystem.
"""

from typing import Dict, List, Set, Tuple


class InteractionRegistry:
    """
    Registry for organism interactions in the ecosystem.
    
    This class manages which interactions are possible between different
    organism types and under what conditions they can occur.
    """
    
    def __init__(self):
        """Initialize the interaction registry."""
        # Mapping from (organism_type1, organism_type2) to list of interaction types
        self.interactions: Dict[Tuple[str, str], List[str]] = {}
        
        # Register default interactions
        self._register_default_interactions()
    
    def _register_default_interactions(self):
        """Register the default ecosystem interactions."""
        # Plant and Herbivore interactions
        self.register_interaction("Plant", "Herbivore", "feeding")
        
        # Plant and Decomposer interactions
        self.register_interaction("Plant", "Decomposer", "symbiosis")
        
        # Herbivore interactions
        self.register_interaction("Herbivore", "Herbivore", "competition")
        
        # Decomposer interactions
        self.register_interaction("Decomposer", "Plant", "decomposition", bidirectional=False)
        self.register_interaction("Decomposer", "Herbivore", "decomposition", bidirectional=False)
        self.register_interaction("Decomposer", "Decomposer", "cooperation", bidirectional=True)
    
    def register_interaction(
        self,
        organism_type1: str,
        organism_type2: str,
        interaction_type: str,
        bidirectional: bool = False
    ) -> None:
        """
        Register an interaction between two organism types.
        
        Args:
            organism_type1: First organism type
            organism_type2: Second organism type
            interaction_type: Type of interaction possible
            bidirectional: If True, the interaction works both ways
        """
        # Create key
        key = (organism_type1, organism_type2)
        
        # Add interaction
        if key not in self.interactions:
            self.interactions[key] = []
        
        if interaction_type not in self.interactions[key]:
            self.interactions[key].append(interaction_type)
        
        # If bidirectional, add reverse direction too
        if bidirectional and organism_type1 != organism_type2:
            reverse_key = (organism_type2, organism_type1)
            
            if reverse_key not in self.interactions:
                self.interactions[reverse_key] = []
            
            if interaction_type not in self.interactions[reverse_key]:
                self.interactions[reverse_key].append(interaction_type)
    
    def unregister_interaction(
        self,
        organism_type1: str,
        organism_type2: str,
        interaction_type: str = None
    ) -> None:
        """
        Remove an interaction between two organism types.
        
        Args:
            organism_type1: First organism type
            organism_type2: Second organism type
            interaction_type: Type of interaction to remove, or None for all
        """
        key = (organism_type1, organism_type2)
        
        if key not in self.interactions:
            return
        
        if interaction_type is None:
            # Remove all interactions
            del self.interactions[key]
        else:
            # Remove specific interaction
            if interaction_type in self.interactions[key]:
                self.interactions[key].remove(interaction_type)
                
                # Cleanup if empty
                if not self.interactions[key]:
                    del self.interactions[key]
    
    def get_interaction_types(
        self,
        organism_type1: str,
        organism_type2: str
    ) -> List[str]:
        """
        Get all possible interaction types between two organism types.
        
        Args:
            organism_type1: First organism type
            organism_type2: Second organism type
            
        Returns:
            List of possible interaction types
        """
        key = (organism_type1, organism_type2)
        
        return self.interactions.get(key, [])
    
    def get_interactions_for_type(self, organism_type: str) -> Dict[str, List[str]]:
        """
        Get all interactions involving a specific organism type.
        
        Args:
            organism_type: Organism type to check
            
        Returns:
            Dictionary of target organism types to interaction types
        """
        result = {}
        
        for (type1, type2), interactions in self.interactions.items():
            if type1 == organism_type:
                result[type2] = interactions
        
        return result


class PredatorPreyInteraction:
    """
    Special interaction class for predator-prey dynamics.
    
    This class manages the complicated rules and calculations involved
    in predator-prey interactions, including hunt success probability,
    energy transfer, and ecological balance.
    """
    
    def __init__(
        self,
        predator_advantage: float = 0.6,
        prey_escape_chance: float = 0.4,
        energy_transfer_efficiency: float = 0.2
    ):
        """
        Initialize predator-prey interaction parameters.
        
        Args:
            predator_advantage: Base hunting success factor (0-1)
            prey_escape_chance: Base prey escape probability (0-1)
            energy_transfer_efficiency: Efficiency of energy transfer from prey to predator
        """
        self.predator_advantage = predator_advantage
        self.prey_escape_chance = prey_escape_chance
        self.energy_transfer_efficiency = energy_transfer_efficiency
    
    def calculate_hunt_success(
        self,
        predator_stats: Dict[str, float],
        prey_stats: Dict[str, float],
        environmental_factors: Dict[str, float]
    ) -> float:
        """
        Calculate the probability of a successful hunt.
        
        Args:
            predator_stats: Predator statistics (speed, strength, etc.)
            prey_stats: Prey statistics (speed, agility, etc.)
            environmental_factors: Current environmental conditions
            
        Returns:
            Probability of hunt success (0-1)
        """
        # Base success chance
        base_chance = self.predator_advantage
        
        # Predator factors
        predator_speed = predator_stats.get("speed", 0.5)
        predator_energy = predator_stats.get("energy", 0.5)
        predator_size = predator_stats.get("size", 1.0)
        
        # Prey factors
        prey_speed = prey_stats.get("speed", 0.5)
        prey_energy = prey_stats.get("energy", 0.5)
        prey_size = prey_stats.get("size", 1.0)
        
        # Environmental factors
        visibility = environmental_factors.get("light_level", 0.5)
        terrain_complexity = environmental_factors.get("terrain_complexity", 0.5)
        
        # Speed advantage
        speed_ratio = predator_speed / max(0.1, prey_speed)
        speed_factor = 0.7 + 0.3 * (speed_ratio - 1)
        
        # Size difference
        size_ratio = predator_size / max(0.1, prey_size)
        size_factor = 0.5 + 0.5 * (size_ratio - 1)
        
        # Energy levels
        energy_factor = predator_energy / max(0.1, prey_energy)
        
        # Environmental modifiers
        # - Higher visibility helps predator
        # - Complex terrain helps prey hide
        environment_factor = visibility / max(0.1, terrain_complexity)
        
        # Combine factors with weights
        success_probability = (
            base_chance * 0.4 +  # Base chance
            speed_factor * 0.25 +  # Speed comparison
            size_factor * 0.15 +  # Size comparison
            energy_factor * 0.1 +  # Energy comparison
            environment_factor * 0.1  # Environmental conditions
        )
        
        # Clamp to valid range
        return max(0.05, min(0.95, success_probability))
    
    def calculate_energy_transfer(
        self,
        prey_energy: float,
        prey_size: float
    ) -> float:
        """
        Calculate how much energy a predator gains from consuming prey.
        
        Args:
            prey_energy: Current energy level of prey
            prey_size: Size of the prey
            
        Returns:
            Energy gained by predator
        """
        # Base energy is proportional to prey's current energy and size
        base_energy = prey_energy * prey_size
        
        # Apply transfer efficiency - only a fraction becomes usable energy
        gained_energy = base_energy * self.energy_transfer_efficiency
        
        return gained_energy
