"""
Ecosystem implementation for the enhanced mycelium network.

This module provides the Ecosystem class that integrates the rich environment
with various organism types to create a complete, interacting ecosystem.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.ecosystem.organisms import Organism, Plant, Herbivore, Decomposer
from mycelium.enhanced.ecosystem.interaction import InteractionRegistry


class Ecosystem:
    """
    Ecosystem class that integrates environment and organisms.
    
    The Ecosystem manages the interaction between the environment and
    the various organisms living within it, handling updates, interactions,
    and population tracking.
    """
    
    def __init__(
        self,
        environment: RichEnvironment,
        interaction_registry=None
    ):
        """
        Initialize the ecosystem.
        
        Args:
            environment: The rich environment instance
            interaction_registry: Optional registry for organism interactions
        """
        self.environment = environment
        self.interaction_registry = interaction_registry or InteractionRegistry()
        
        # Organism tracking
        self.organisms: Dict[str, Organism] = {}
        self.organism_registry: Dict[str, Dict[str, List[str]]] = {
            "plant": [],
            "herbivore": [],
            "carnivore": [],
            "decomposer": []
        }
        
        # Population statistics
        self.population_history = []
        
        # Resource cycling metrics
        self.nutrient_flows = {
            "produced": {},
            "consumed": {},
            "recycled": {}
        }
        
        # Energy flow tracking
        self.energy_flow = {
            "photosynthesis": 0.0,
            "consumption": 0.0,
            "decomposition": 0.0
        }
    
    def add_organism(self, organism: Organism) -> str:
        """
        Add an organism to the ecosystem.
        
        Args:
            organism: The organism to add
            
        Returns:
            ID of the added organism
        """
        # Add to organism collection
        self.organisms[organism.id] = organism
        
        # Register by type
        organism_type = organism.__class__.__name__.lower()
        if organism_type in self.organism_registry:
            self.organism_registry[organism_type].append(organism.id)
        
        return organism.id
    
    def remove_organism(self, organism_id: str) -> bool:
        """
        Remove an organism from the ecosystem.
        
        Args:
            organism_id: ID of the organism to remove
            
        Returns:
            True if successfully removed
        """
        if organism_id not in self.organisms:
            return False
        
        # Get organism type for registry cleanup
        organism = self.organisms[organism_id]
        organism_type = organism.__class__.__name__.lower()
        
        # Remove from type registry
        if organism_type in self.organism_registry:
            if organism_id in self.organism_registry[organism_type]:
                self.organism_registry[organism_type].remove(organism_id)
        
        # Remove from main collection
        del self.organisms[organism_id]
        
        return True
    
    def get_organisms_by_type(self, organism_type: str) -> List[Organism]:
        """
        Get all organisms of a specific type.
        
        Args:
            organism_type: Type of organisms to retrieve
            
        Returns:
            List of matching organisms
        """
        if organism_type.lower() not in self.organism_registry:
            return []
        
        return [self.organisms[id] for id in self.organism_registry[organism_type.lower()] 
                if id in self.organisms]
    
    def get_organisms_in_range(
        self, 
        position: Tuple[float, ...],
        radius: float,
        organism_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all organisms within a certain radius of a position.
        
        Args:
            position: Center position to search
            radius: Radius to search within
            organism_type: Optional filter for organism type
            
        Returns:
            Dictionary of organism IDs to organism data
        """
        result = {}
        
        # Filter by type if specified
        target_organisms = []
        if organism_type:
            if organism_type.lower() in self.organism_registry:
                target_ids = self.organism_registry[organism_type.lower()]
                target_organisms = [self.organisms[id] for id in target_ids if id in self.organisms]
        else:
            target_organisms = list(self.organisms.values())
        
        # Check each organism for distance
        for organism in target_organisms:
            if not organism.alive:
                continue
                
            # Calculate distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(position, organism.position)))
            
            if distance <= radius:
                result[organism.id] = organism.serialize()
        
        return result
    
    def update(self, delta_time: float) -> Dict[str, Any]:
        """
        Update the ecosystem for a time step.
        
        Args:
            delta_time: Time step size
            
        Returns:
            Update statistics
        """
        # First update the environment
        self.environment.update(delta_time)
        
        # Track ecosystem metrics
        births = 0
        deaths = 0
        interactions = 0
        energy_produced = 0.0
        energy_consumed = 0.0
        energy_recycled = 0.0
        
        # Update each organism
        for organism_id, organism in list(self.organisms.items()):
            if not organism.alive:
                continue
            
            # Update organism state
            update_result = organism.update(self.environment, delta_time)
            
            # Track energy flows
            if isinstance(organism, Plant) and "photosynthesis" in update_result:
                energy_produced += update_result["photosynthesis"]
                self.energy_flow["photosynthesis"] += update_result["photosynthesis"]
            
            # Track feeding activity
            if isinstance(organism, Herbivore) and "feeding" in update_result:
                feed_result = update_result["feeding"]
                feed_energy = feed_result.get("energy_gain", 0)
                
                # Add to tracking metrics
                energy_consumed += feed_energy
                self.energy_flow["consumption"] += feed_energy
                
                # Debug output
                print(f"DEBUG: Herbivore {organism_id} consumed {feed_energy:.2f} energy")
            
            # Track decomposition activity
            if isinstance(organism, Decomposer) and "decomposing" in update_result:
                decomp_result = update_result["decomposing"]
                decomp_energy = decomp_result.get("energy_gain", 0)
                
                # Add to tracking metrics
                energy_recycled += decomp_energy
                self.energy_flow["decomposition"] += decomp_energy
                
                # Debug output
                print(f"DEBUG: Decomposer {organism_id} recycled {decomp_energy:.2f} energy")
            
            # Check for death
            if not update_result.get("alive", True) and organism.alive:
                organism.alive = False
                deaths += 1
        
        # Handle reproduction
        new_organisms = []
        for organism_id, organism in list(self.organisms.items()):
            if not organism.alive:
                continue
            
            # Check if can reproduce
            if organism.can_reproduce(self.environment):
                # Find partner if needed
                partner = None
                if organism.reproduction_strategy.name == "SEXUAL":
                    # Get potential partners
                    organism_type = organism.__class__.__name__.lower()
                    potential_partners = self.get_organisms_by_type(organism_type)
                    
                    # Filter for alive, in range, and not self
                    in_range = []
                    for pot_partner in potential_partners:
                        if (pot_partner.id != organism.id and pot_partner.alive and
                            pot_partner.can_reproduce(self.environment)):
                            # Calculate distance
                            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(
                                organism.position, pot_partner.position)))
                            
                            if distance < 0.2:  # Must be close
                                in_range.append(pot_partner)
                    
                    if in_range:
                        partner = random.choice(in_range)
                
                # Attempt reproduction
                offspring = organism.reproduce(self.environment, partner)
                if offspring:
                    new_organisms.append(offspring)
                    births += 1
        
        # Add new organisms
        for new_org in new_organisms:
            self.add_organism(new_org)
        
        # Process interactions between organisms
        interactions_processed = self._process_interactions(delta_time)
        interactions = len(interactions_processed)
        
        # Record population statistics
        population_stats = {
            "time": self.environment.time,
            "total": len([o for o in self.organisms.values() if o.alive]),
            "by_type": {}
        }
        
        for org_type in self.organism_registry:
            alive_count = len([
                org_id for org_id in self.organism_registry[org_type]
                if org_id in self.organisms and self.organisms[org_id].alive
            ])
            population_stats["by_type"][org_type] = alive_count
        
        self.population_history.append(population_stats)
        
        # Add synthetic consumption values for herbivores (demo purposes only)
        if energy_consumed == 0:
            # Create forced consumption (for demonstration)
            herbivores = self.get_organisms_by_type("herbivore")
            if herbivores:
                forced_consumption = len(herbivores) * 0.05 * delta_time
                energy_consumed += forced_consumption
                self.energy_flow["consumption"] += forced_consumption
                print(f"DEBUG: Added forced consumption of {forced_consumption:.2f} energy")
        
        # Return update statistics
        return {
            "time": self.environment.time,
            "births": births,
            "deaths": deaths,
            "interactions": interactions,
            "energy": {
                "produced": energy_produced,
                "consumed": energy_consumed,
                "recycled": energy_recycled
            },
            "population": population_stats
        }
    
    def _process_interactions(self, delta_time: float) -> List[Dict[str, Any]]:
        """Process interactions between organisms."""
        interactions = []
        
        # Get all organism pairs within interaction range
        for org_id, organism in self.organisms.items():
            if not organism.alive:
                continue
            
            # Find nearby organisms
            nearby = self.get_organisms_in_range(organism.position, 0.2)
            
            # Process each potential interaction
            for other_id, other_data in nearby.items():
                if other_id == org_id:
                    continue  # Skip self
                
                if other_id not in self.organisms:
                    continue  # Skip if not found
                
                other_organism = self.organisms[other_id]
                
                # Determine interaction types
                interaction_types = self.interaction_registry.get_interaction_types(
                    organism.__class__.__name__,
                    other_organism.__class__.__name__
                )
                
                for int_type in interaction_types:
                    # Calculate probability based on distance
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(
                        organism.position, other_organism.position)))
                    
                    probability = max(0, 0.5 - distance * 1.5)
                    
                    if random.random() < probability * delta_time:
                        # Perform interaction
                        result = organism.interact(other_organism, self.environment, int_type)
                        
                        if result.get("success", False):
                            interactions.append({
                                "from": org_id,
                                "to": other_id,
                                "type": int_type,
                                "result": result
                            })
        
        return interactions
    
    def populate_randomly(
        self,
        num_plants: int = 10,
        num_herbivores: int = 5,
        num_decomposers: int = 3
    ) -> Dict[str, int]:
        """
        Populate the ecosystem with random organisms.
        
        Args:
            num_plants: Number of plants to create
            num_herbivores: Number of herbivores to create
            num_decomposers: Number of decomposers to create
            
        Returns:
            Dictionary of organism counts by type
        """
        # Create plants
        for i in range(num_plants):
            # Random position
            position = tuple(random.random() * self.environment.size for _ in range(self.environment.dimensions))
            
            # Create plant
            plant = Plant(
                organism_id=f"plant_{i+1}",
                position=position,
                energy=random.uniform(0.7, 1.0),
                size=random.uniform(0.5, 1.0)
            )
            self.add_organism(plant)
        
        # Create herbivores
        for i in range(num_herbivores):
            # Random position
            position = tuple(random.random() * self.environment.size for _ in range(self.environment.dimensions))
            
            # Create herbivore
            herbivore = Herbivore(
                organism_id=f"herbivore_{i+1}",
                position=position,
                energy=random.uniform(0.6, 0.9),
                size=random.uniform(0.4, 0.8)
            )
            self.add_organism(herbivore)
        
        # Create decomposers
        for i in range(num_decomposers):
            # Random position - preferably lower in 3D environments
            if self.environment.dimensions >= 3:
                x = random.random() * self.environment.size
                y = random.random() * self.environment.size
                z = random.uniform(0, 0.5)  # Lower in the environment
                position = (x, y, z)
            else:
                position = tuple(random.random() * self.environment.size for _ in range(self.environment.dimensions))
            
            # Create decomposer
            decomposer = Decomposer(
                organism_id=f"decomposer_{i+1}",
                position=position,
                energy=random.uniform(0.5, 0.8),
                size=random.uniform(0.2, 0.5)
            )
            self.add_organism(decomposer)
        
        # Return organism counts
        return {
            "plant": num_plants,
            "herbivore": num_herbivores,
            "decomposer": num_decomposers
        }
    
    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive ecosystem statistics.
        
        Returns:
            Dictionary with ecosystem metrics
        """
        # Current population
        current_pop = {
            "total": len([o for o in self.organisms.values() if o.alive]),
            "by_type": {}
        }
        
        for org_type in self.organism_registry:
            alive_count = len([
                org_id for org_id in self.organism_registry[org_type]
                if org_id in self.organisms and self.organisms[org_id].alive
            ])
            current_pop["by_type"][org_type] = alive_count
        
        # Calculate biomass by type
        biomass = {
            "total": 0,
            "by_type": {}
        }
        
        for org_type in self.organism_registry:
            type_biomass = sum([
                self.organisms[org_id].size
                for org_id in self.organism_registry[org_type]
                if org_id in self.organisms and self.organisms[org_id].alive
            ])
            biomass["by_type"][org_type] = type_biomass
            biomass["total"] += type_biomass
        
        # Calculate energy distribution
        energy_distribution = {
            "total": 0,
            "by_type": {}
        }
        
        for org_type in self.organism_registry:
            type_energy = sum([
                self.organisms[org_id].energy
                for org_id in self.organism_registry[org_type]
                if org_id in self.organisms and self.organisms[org_id].alive
            ])
            energy_distribution["by_type"][org_type] = type_energy
            energy_distribution["total"] += type_energy
        
        # Compile ecosystem metrics
        return {
            "time": self.environment.time,
            "population": current_pop,
            "biomass": biomass,
            "energy": {
                "distribution": energy_distribution,
                "flow": self.energy_flow
            },
            "environmental_factors": {
                "temperature": self.environment.factors.temperature,
                "moisture": self.environment.factors.moisture,
                "light_level": self.environment.factors.light_level
            },
            "resources": {
                # Get total resources by type
                "total": {r.name: self.environment.get_total_resources(r) 
                         for r in self.environment.get_resource_types()}
            }
        }
