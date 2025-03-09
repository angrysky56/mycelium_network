"""
Base Organism class for the enhanced ecosystem.

This module defines the abstract base class that all organisms will inherit from.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from mycelium.enhanced.ecosystem.enums import NutrientNeed, ReproductionStrategy


class Organism(ABC):
    """
    Base class for all organisms in the ecosystem.
    
    This abstract class defines the common interface and properties
    for all organisms, regardless of their specific type.
    """
    
    def __init__(
        self,
        organism_id: str,
        position: Tuple[float, ...],
        energy: float = 1.0,
        size: float = 1.0,
        lifespan: float = 100.0,
        reproduction_rate: float = 0.01,
        reproduction_strategy: ReproductionStrategy = ReproductionStrategy.ASEXUAL,
        properties: Dict[str, Any] = None
    ):
        """
        Initialize an organism.
        
        Args:
            organism_id: Unique identifier for this organism
            position: Spatial position of the organism
            energy: Current energy level (0-1)
            size: Size of the organism
            lifespan: Maximum lifespan in time units
            reproduction_rate: Base probability of reproduction per time unit
            reproduction_strategy: How this organism reproduces
            properties: Additional properties specific to this organism
        """
        self.id = organism_id
        self.position = position
        self.energy = energy
        self.size = size
        self.alive = True
        self.age = 0.0
        self.lifespan = lifespan
        self.reproduction_rate = reproduction_rate
        self.reproduction_strategy = reproduction_strategy
        self.properties = properties or {}
        
        # Metabolism
        self.metabolism_rate = 0.01  # Base energy consumption per time unit
        self.nutrient_needs = {}     # Resource type to necessity level
        
        # Interaction history
        self.interaction_history = []
    
    @abstractmethod
    def update(self, environment, delta_time: float) -> Dict[str, Any]:
        """
        Update the organism state for a time step.
        
        Args:
            environment: The environment the organism exists in
            delta_time: Time step size
            
        Returns:
            State changes from this update
        """
        # Base update logic for all organisms
        self.age += delta_time
        
        # Consume base metabolism energy
        self.energy -= self.metabolism_rate * delta_time
        
        # Check for death by age or starvation
        if self.age >= self.lifespan or self.energy <= 0:
            self.alive = False
            return {"alive": False, "cause": "age" if self.age >= self.lifespan else "starvation"}
        
        return {"alive": True}
    
    @abstractmethod
    def interact(self, other_organism, environment, interaction_type: str) -> Dict[str, Any]:
        """
        Interact with another organism.
        
        Args:
            other_organism: The organism to interact with
            environment: The environment context
            interaction_type: Type of interaction to perform
            
        Returns:
            Interaction results
        """
        pass
    
    def can_reproduce(self, environment) -> bool:
        """
        Check if this organism can reproduce.
        
        Args:
            environment: The environment context
            
        Returns:
            True if reproduction is possible
        """
        # Base checks applicable to all organisms
        if not self.alive:
            return False
        
        if self.energy < 0.6:  # Need sufficient energy
            return False
        
        if self.age < self.lifespan * 0.1:  # Too young
            return False
        
        if self.age > self.lifespan * 0.8:  # Too old
            return False
        
        # Random chance based on reproduction rate
        return random.random() < self.reproduction_rate
    
    @abstractmethod
    def reproduce(self, environment, partner=None) -> Optional['Organism']:
        """
        Create a new organism through reproduction.
        
        Args:
            environment: The environment context
            partner: Optional partner for sexual reproduction
            
        Returns:
            New organism or None if reproduction failed
        """
        pass
    
    def serialize(self) -> Dict[str, Any]:
        """
        Convert organism state to serializable dictionary.
        
        Returns:
            Dictionary representation of organism
        """
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "position": self.position,
            "energy": self.energy,
            "size": self.size,
            "alive": self.alive,
            "age": self.age,
            "lifespan": self.lifespan,
            "properties": self.properties
        }
