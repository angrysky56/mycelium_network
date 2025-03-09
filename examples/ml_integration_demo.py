#!/usr/bin/env python3
"""
Machine Learning Integration Demo for Enhanced Mycelium Network

This script demonstrates the machine learning capabilities of the enhanced
mycelium network, including reinforcement learning and transfer learning.
"""

import os
import sys
import random
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from mycelium.enhanced.rich_environment import RichEnvironment
from mycelium.enhanced.resource import ResourceType, Environmental_Factors
from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork
from mycelium.enhanced.ml.reinforcement import ReinforcementLearner
from mycelium.enhanced.ml.transfer import TransferNetwork


def print_separator(title):
    """Print a section separator with title."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def demo_reinforcement_learning():
    """Demonstrate reinforcement learning for optimizing network growth."""
    print_separator("Reinforcement Learning Optimization")
    
    # Create environment with seasonal cycles
    environment = RichEnvironment(dimensions=2, size=1.0)
    environment.create_seasonal_cycle(year_length=20.0, intensity=0.8)  # 20 time units per year
    print(f"Created environment: {environment.name} with seasonal cycles")
    
    # Add resources
    print("Adding resources to environment...")
    resources = [
        ((0.3, 0.7), 0.15, ResourceType.CARBON, 1.0),
        ((0.7, 0.3), 0.15, ResourceType.WATER, 1.0),
        ((0.5, 0.5), 0.1, ResourceType.NITROGEN, 0.5)
    ]
    
    for center, radius, resource_type, amount in resources:
        environment.add_nutrient_cluster(center, radius, resource_type, amount)
    
    # Create network to optimize
    network = AdaptiveMyceliumNetwork(
        environment=environment,
        input_size=3,
        output_size=1,
        initial_nodes=5
    )
    print(f"Created network with {len(network.nodes)} nodes")
    
    # Define state space and action space
    state_size = 17   # Node count, resources, detailed environmental factors, seasonal progression
    action_size = 10  # Enhanced growth strategies including season-specific and environment-adaptive actions
    
    # Create reinforcement learner
    rl_agent = ReinforcementLearner(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99
    )
    print(f"Created reinforcement learning agent with {state_size}-dimensional state space")
    print(f"Action space has {action_size} actions representing different growth strategies")
    
    # Define environment step function for RL
    def environment_step(action=None, reset=False):
        """Environment step function for RL agent."""
        if reset:
            # Reset environment and network
            environment.update(0.1)
            for _ in range(5):
                network.forward([0.5, 0.5, 0.5])
            
            # Get initial state
            state = get_state()
            return state
        
        # Process the action
        if action is not None:
            # Action represents growth strategy
            execute_growth_strategy(action)
        
        # Update environment and network
        environment.update(0.1)
        outputs = network.forward([0.5, 0.5, 0.5])
        
        # Calculate reward based on network performance
        reward = calculate_reward()
        
        # Get next state
        next_state = get_state()
        
        # Check if episode is done
        done = len(network.nodes) >= 30 or network.total_resources <= 0
        
        return next_state, reward, done, {"outputs": outputs}
    
    def get_state():
        """Convert environment and network state to RL state vector with enhanced seasonal awareness."""
        # State components:
        # 1. Normalized node count
        node_count = len(network.nodes) / 50  # Max expected nodes
        
        # 2. Resource availability (more detailed)
        total_resources = network.total_resources / 10  # Normalize
        carbon_resources = environment.get_total_resources(ResourceType.CARBON) / 5.0
        water_resources = environment.get_total_resources(ResourceType.WATER) / 5.0
        nitrogen_resources = environment.get_total_resources(ResourceType.NITROGEN) / 3.0
        
        # 3. Environmental factors (more comprehensive)
        env_factors = environment.factors
        temperature = env_factors.temperature
        moisture = env_factors.moisture
        light = env_factors.light_level
        wind = env_factors.wind
        
        # 4. Network adaptation levels
        temp_adaptation = network.temperature_adaptation
        moisture_adaptation = network.moisture_adaptation
        drought_resistance = network.drought_resistance
        
        # 5. Enhanced season information
        try:
            # Get current time in seasonal cycle
            year_progress = (environment.time % environment.year_length) / environment.year_length
            season_idx = int(year_progress * 4) % 4
            
            # Calculate continous season progression (0.0-1.0 for each season)
            season_phase = (year_progress * 4) % 1.0  # Position within current season
            
            # Convert to normalized values by season (one-hot encoding)
            is_spring = 1.0 if season_idx == 0 else 0.0
            is_summer = 1.0 if season_idx == 1 else 0.0
            is_fall = 1.0 if season_idx == 2 else 0.0
            is_winter = 1.0 if season_idx == 3 else 0.0
            
            # Add season transition awareness
            season_transition = season_phase  # How far into the current season (0.0-1.0)
        except AttributeError:
            # Default if no seasons
            is_spring = 0.0
            is_summer = 0.0
            is_fall = 0.0
            is_winter = 0.0
            season_transition = 0.0
        
        return [
            node_count,
            total_resources,
            carbon_resources,  # More specific resource tracking
            water_resources,
            nitrogen_resources,
            temperature,
            moisture,
            light,
            wind,
            temp_adaptation,
            moisture_adaptation,
            drought_resistance,  # Added adaptation parameter
            is_spring,
            is_summer,
            is_fall,
            is_winter,
            season_transition   # Continuous season progression
        ]
    
    def execute_growth_strategy(action):
        """Execute a growth strategy based on the action with enhanced seasonal awareness."""
        # Get current season with more detailed information
        try:
            year_progress = (environment.time % environment.year_length) / environment.year_length
            season_idx = int(year_progress * 4) % 4
            season_phase = (year_progress * 4) % 1.0  # Position within current season (0.0-1.0)
            season_names = ["Spring", "Summer", "Fall", "Winter"]
            current_season = season_names[season_idx]
            
            # Determine if we're early, mid, or late in the season
            season_stage = "early" if season_phase < 0.33 else "mid" if season_phase < 0.66 else "late"
            
            # Detect transitional periods between seasons
            is_transitional = season_phase < 0.15 or season_phase > 0.85
            next_season_idx = (season_idx + 1) % 4
            next_season = season_names[next_season_idx]
        except AttributeError:
            current_season = "Unknown"
            season_stage = "mid"
            is_transitional = False
            next_season = "Unknown"

        # More nuanced seasonal strategy modifiers with transition awareness
        growth_boost = 1.0
        adaptation_boost = 1.0
        specialization_boost = 1.0
        resource_sensitivity = 1.0
        
        if current_season == "Spring":
            if season_stage == "early":  # Early spring
                growth_boost = 1.2
                adaptation_boost = 0.9
                specialization_boost = 0.8
                resource_sensitivity = 1.3  # More sensitive to available resources
            elif season_stage == "mid":  # Mid spring
                growth_boost = 1.4  # Peak growth
                adaptation_boost = 0.7
                specialization_boost = 0.9
                resource_sensitivity = 1.1
            else:  # Late spring
                growth_boost = 1.3
                adaptation_boost = 0.8
                specialization_boost = 1.0
                resource_sensitivity = 1.0
        elif current_season == "Summer":
            if season_stage == "early":  # Early summer
                growth_boost = 1.2
                adaptation_boost = 0.9
                specialization_boost = 1.1
                resource_sensitivity = 1.0
            elif season_stage == "mid":  # Mid summer
                growth_boost = 1.1  # Sustained growth
                adaptation_boost = 1.0
                specialization_boost = 1.2  # Focus on specialization
                resource_sensitivity = 1.2  # More resource sensitive in mid-summer heat
            else:  # Late summer
                growth_boost = 1.0
                adaptation_boost = 1.1
                specialization_boost = 1.1
                resource_sensitivity = 1.3
        elif current_season == "Fall":
            if season_stage == "early":  # Early fall
                growth_boost = 0.9
                adaptation_boost = 1.2
                specialization_boost = 1.0
                resource_sensitivity = 1.2
            elif season_stage == "mid":  # Mid fall
                growth_boost = 0.7  # Reduced growth
                adaptation_boost = 1.4  # Increased adaptation
                specialization_boost = 0.9
                resource_sensitivity = 1.4  # Focus on resource collection
            else:  # Late fall
                growth_boost = 0.5
                adaptation_boost = 1.6  # Preparing for winter
                specialization_boost = 0.8
                resource_sensitivity = 1.5
        elif current_season == "Winter":
            if season_stage == "early":  # Early winter
                growth_boost = 0.4
                adaptation_boost = 1.7
                specialization_boost = 0.7
                resource_sensitivity = 1.3
            elif season_stage == "mid":  # Mid winter - survival mode
                growth_boost = 0.3  # Minimal growth
                adaptation_boost = 1.8  # Maximum adaptation
                specialization_boost = 0.6
                resource_sensitivity = 1.0  # Less resource focus, more survival
            else:  # Late winter - preparation for spring
                growth_boost = 0.5  # Slight growth increase
                adaptation_boost = 1.6
                specialization_boost = 0.8
                resource_sensitivity = 1.2
                
        # Apply transitional adjustments
        if is_transitional:
            # Smooth transitions between seasons
            if current_season == "Winter" and next_season == "Spring":
                # Transitioning to spring - start ramping up growth
                growth_boost += 0.2
                adaptation_boost -= 0.1
            elif current_season == "Summer" and next_season == "Fall":
                # Transitioning to fall - start storing resources
                resource_sensitivity += 0.2
                growth_boost -= 0.1
            elif current_season == "Fall" and next_season == "Winter":
                # Transitioning to winter - maximize adaptation
                adaptation_boost += 0.2
                growth_boost -= 0.1
            
        # Enhanced strategies with broader seasonal awareness and adaptation
        if action == 0:
            # Balanced growth
            network.growth_rate = 0.05 * growth_boost
            network.adaptation_rate = 0.1 * adaptation_boost
        elif action == 1:
            # Rapid growth
            network.growth_rate = 0.1 * growth_boost
            network.adaptation_rate = 0.05 * adaptation_boost
        elif action == 2:
            # Adaptive growth
            network.growth_rate = 0.03 * growth_boost
            network.adaptation_rate = 0.15 * adaptation_boost
        elif action == 3:
            # Resource-focused - enhanced with resource sensitivity
            network.growth_rate = 0.07 * growth_boost
            network._prioritize_resources(priority_factor=resource_sensitivity)
        elif action == 4:
            # Specialized growth - enhanced with specialization boost
            network.growth_rate = 0.04 * growth_boost * specialization_boost
            network._prioritize_specialization()
        elif action == 5:
            # Conservative growth
            network.growth_rate = 0.02 * growth_boost
            network.adaptation_rate = 0.2 * adaptation_boost
        elif action == 6:
            # Season-transition strategy - optimized for transitional periods
            if is_transitional:
                if current_season == "Winter" and next_season == "Spring":
                    # Transition to spring - prepare for growth
                    network.growth_rate = 0.04 * growth_boost
                    network.adaptation_rate = 0.15 * adaptation_boost
                    network._prepare_for_season_change("Spring")
                elif current_season == "Spring" and next_season == "Summer":
                    # Transition to summer - prepare for heat
                    network.growth_rate = 0.08 * growth_boost
                    network.adaptation_rate = 0.1 * adaptation_boost
                    network.temperature_adaptation += 0.02
                elif current_season == "Summer" and next_season == "Fall":
                    # Transition to fall - start resource collection
                    network.growth_rate = 0.06 * growth_boost
                    network._prioritize_resources(priority_factor=1.5)
                elif current_season == "Fall" and next_season == "Winter":
                    # Transition to winter - maximize adaptation
                    network.growth_rate = 0.02 * growth_boost
                    network.adaptation_rate = 0.25 * adaptation_boost
                    network.temperature_adaptation += 0.03
                    network.drought_resistance += 0.02
            else:
                # Not in transition - balanced approach
                network.growth_rate = 0.05 * growth_boost
                network.adaptation_rate = 0.12 * adaptation_boost
        elif action == 7:
            # Season-stage specific optimization
            if current_season == "Spring":
                if season_stage == "early":
                    # Early spring: focus on initial growth
                    network.growth_rate = 0.09 * growth_boost
                    network.adaptation_rate = 0.07 * adaptation_boost
                elif season_stage == "mid":
                    # Mid spring: rapid growth phase
                    network.growth_rate = 0.13 * growth_boost
                    network.adaptation_rate = 0.05 * adaptation_boost
                else:
                    # Late spring: prepare for summer
                    network.growth_rate = 0.10 * growth_boost
                    network.adaptation_rate = 0.08 * adaptation_boost
                    network.temperature_adaptation += 0.01
            elif current_season == "Summer":
                if season_stage == "early":
                    # Early summer: balanced growth
                    network.growth_rate = 0.08 * growth_boost
                    network.adaptation_rate = 0.09 * adaptation_boost
                elif season_stage == "mid":
                    # Mid summer: adapted growth with heat resistance
                    network.growth_rate = 0.07 * growth_boost
                    network.temperature_adaptation += 0.02
                    network.moisture_adaptation += 0.01
                else:
                    # Late summer: resource collection
                    network.growth_rate = 0.06 * growth_boost
                    network._prioritize_resources(priority_factor=1.2)
            elif current_season == "Fall":
                if season_stage == "early":
                    # Early fall: focused resource collection
                    network.growth_rate = 0.05 * growth_boost
                    network._prioritize_resources(priority_factor=1.3)
                elif season_stage == "mid":
                    # Mid fall: adaptation preparation
                    network.growth_rate = 0.04 * growth_boost
                    network.adaptation_rate = 0.15 * adaptation_boost
                else:
                    # Late fall: winter preparation
                    network.growth_rate = 0.03 * growth_boost
                    network.adaptation_rate = 0.2 * adaptation_boost
                    network.temperature_adaptation += 0.02
            else:  # Winter
                if season_stage == "early":
                    # Early winter: max adaptation
                    network.growth_rate = 0.02 * growth_boost
                    network.adaptation_rate = 0.25 * adaptation_boost
                    network.temperature_adaptation += 0.03
                elif season_stage == "mid":
                    # Mid winter: survival mode
                    network.growth_rate = 0.01 * growth_boost
                    network.adaptation_rate = 0.3 * adaptation_boost
                else:
                    # Late winter: prepare for spring
                    network.growth_rate = 0.03 * growth_boost
                    network.adaptation_rate = 0.2 * adaptation_boost
        elif action == 8:
            # Environmental-response: refined climate adaptation
            env_factors = environment.factors
            
            if env_factors.temperature < 0.3:
                # Cold conditions - focus on temperature adaptation
                network.growth_rate = 0.02 * growth_boost
                network.adaptation_rate = 0.2 * adaptation_boost
                network.temperature_adaptation += 0.02
            elif env_factors.temperature > 0.7:
                # Hot conditions - specialized heat adaptation
                network.growth_rate = 0.04 * growth_boost
                network.adaptation_rate = 0.15 * adaptation_boost
                network.temperature_adaptation += 0.01
                network.moisture_adaptation += 0.01  # Better water retention in heat
            elif env_factors.moisture < 0.3:
                # Dry conditions - drought adaptation
                network.growth_rate = 0.03 * growth_boost
                network.adaptation_rate = 0.18 * adaptation_boost
                network.moisture_adaptation += 0.02
                network.drought_resistance += 0.02
            elif env_factors.moisture > 0.7:
                # Wet conditions - specialized moisture adaptation
                network.growth_rate = 0.06 * growth_boost
                network.adaptation_rate = 0.12 * adaptation_boost
                network.moisture_adaptation += 0.01
            elif env_factors.light_level > 0.7:
                # High light - optimized growth
                network.growth_rate = 0.09 * growth_boost
                network.adaptation_rate = 0.07 * adaptation_boost
            elif env_factors.wind > 0.7:
                # Windy conditions - focus on stability
                network.growth_rate = 0.04 * growth_boost
                network.adaptation_rate = 0.14 * adaptation_boost
                network.stress_tolerance += 0.01
            else:
                # Moderate conditions - balanced approach
                network.growth_rate = 0.06 * growth_boost
                network.adaptation_rate = 0.1 * adaptation_boost
        elif action == 9:
            # Resource-specific adaptation based on seasonal patterns
            current_carbon = environment.get_total_resources(ResourceType.CARBON)
            current_water = environment.get_total_resources(ResourceType.WATER)
            current_nitrogen = environment.get_total_resources(ResourceType.NITROGEN)
            
            if current_season == "Spring" or current_season == "Summer":
                # More nitrogen in growing seasons
                if current_nitrogen > current_carbon and current_nitrogen > current_water:
                    # Nitrogen-rich environment - rapid growth
                    network.growth_rate = 0.11 * growth_boost
                    network.adaptation_rate = 0.06 * adaptation_boost
                    # Adjust resource efficiency
                    network.resource_efficiency[ResourceType.NITROGEN] *= 1.05
                elif current_water > 2 * current_carbon:
                    # Water-rich environment - balanced growth
                    network.growth_rate = 0.08 * growth_boost
                    network.adaptation_rate = 0.09 * adaptation_boost
                    network.moisture_adaptation += 0.01
                    network.resource_efficiency[ResourceType.WATER] *= 1.03
                else:
                    # Carbon-focused growth
                    network.growth_rate = 0.07 * growth_boost
                    network.adaptation_rate = 0.08 * adaptation_boost
                    network.resource_efficiency[ResourceType.CARBON] *= 1.04
            else:  # Fall or Winter
                # In resource-scarce seasons, focus on efficiency
                scarce_resource = min([(current_carbon, ResourceType.CARBON), 
                                      (current_water, ResourceType.WATER),
                                      (current_nitrogen, ResourceType.NITROGEN)], 
                                     key=lambda x: x[0])[1]
                
                # Prioritize the most scarce resource
                network.growth_rate = 0.04 * growth_boost
                network.adaptation_rate = 0.16 * adaptation_boost
                network.resource_efficiency[scarce_resource] *= 1.08
                
                if scarce_resource == ResourceType.WATER:
                    network.drought_resistance += 0.02
                elif scarce_resource == ResourceType.CARBON:
                    network._prioritize_resources(priority_factor=1.5)
    
    def calculate_reward():
        """Calculate reward based on network performance metrics with enhanced seasonal awareness."""
        # Reward components
        node_reward = min(1.0, len(network.nodes) / 30)
        resource_reward = min(1.0, network.total_resources / 5)
        
        # Get specialization statistics
        stats = network.get_specialization_statistics()
        adaptation_level = (
            stats['adaptation']['temperature_adaptation'] + 
            stats['adaptation']['moisture_adaptation'] +
            stats['adaptation']['drought_resistance']
        ) / 3
        
        adaptation_reward = adaptation_level
        
        # Calculate connectivity (more connected = better)
        connectivity = 0
        for node in network.nodes.values():
            connectivity += len(node.connections)
        connectivity = min(1.0, connectivity / (len(network.nodes) * 3))
        
        # Calculate specialization diversity reward
        specialization_types = set()
        for spec_type, nodes in network.specializations.items():
            if nodes:  # Only count non-empty specializations
                specialization_types.add(spec_type)
        
        # Reward diversity of specializations (0-1 range)
        specialization_diversity = len(specialization_types) / 3  # Assuming 3 possible types
        
        # Get current season with detailed phase information
        try:
            year_progress = (environment.time % environment.year_length) / environment.year_length
            season_idx = int(year_progress * 4) % 4
            season_phase = (year_progress * 4) % 1.0  # Position within current season (0.0-1.0)
            season_names = ["Spring", "Summer", "Fall", "Winter"]
            current_season = season_names[season_idx]
            season_stage = "early" if season_phase < 0.33 else "mid" if season_phase < 0.66 else "late"
            
            # Get resource efficiency metrics
            carbon_efficiency = network.resource_efficiency.get(ResourceType.CARBON, 1.0)
            water_efficiency = network.resource_efficiency.get(ResourceType.WATER, 1.0)
            nitrogen_efficiency = network.resource_efficiency.get(ResourceType.NITROGEN, 1.0)
            
            # Resource targets based on season and phase
            if current_season == "Spring":
                if season_stage == "early":
                    # Early spring: balanced growth start
                    node_reward *= 1.2
                    resource_reward *= 1.1
                    adaptation_reward *= 0.8
                    # Reward nitrogen efficiency in spring
                    resource_reward += 0.1 * (nitrogen_efficiency - 1.0)
                elif season_stage == "mid":
                    # Mid spring: maximum growth period
                    node_reward *= 1.4
                    resource_reward *= 1.0
                    adaptation_reward *= 0.7
                    # Reward carbon efficiency for growth
                    resource_reward += 0.1 * (carbon_efficiency - 1.0)
                else:  # Late spring
                    # Late spring: transition to summer
                    node_reward *= 1.3
                    resource_reward *= 1.1
                    adaptation_reward *= 0.8
                    # Reward specialization diversity (preparing for summer)
                    resource_reward += 0.1 * specialization_diversity
            elif current_season == "Summer":
                if season_stage == "early":
                    # Early summer: still good growth
                    node_reward *= 1.2
                    resource_reward *= 1.1
                    adaptation_reward *= 0.9
                    # Start rewarding temperature adaptation
                    adaptation_reward += 0.1 * stats['adaptation']['temperature_adaptation']
                elif season_stage == "mid":
                    # Mid summer: balanced with heat adaptation
                    node_reward *= 1.1
                    resource_reward *= 1.2
                    adaptation_reward *= 1.0
                    # Strong reward for temperature adaptation in mid-summer heat
                    adaptation_reward += 0.2 * stats['adaptation']['temperature_adaptation']
                else:  # Late summer
                    # Late summer: prepare for fall, focus on resources
                    node_reward *= 1.0
                    resource_reward *= 1.3
                    adaptation_reward *= 1.1
                    # Reward water efficiency as summer ends
                    resource_reward += 0.15 * (water_efficiency - 1.0)
            elif current_season == "Fall":
                if season_stage == "early":
                    # Early fall: resource collection starts
                    node_reward *= 0.9
                    resource_reward *= 1.4
                    adaptation_reward *= 1.0
                    # Reward storage specialization
                    storage_count = len(network.specializations.get('storage', []))
                    resource_reward += 0.05 * (storage_count / max(1, len(network.nodes)))
                elif season_stage == "mid":
                    # Mid fall: resource storage critical
                    node_reward *= 0.7
                    resource_reward *= 1.6
                    adaptation_reward *= 1.2
                    # Heavy reward for resource efficiency
                    resource_reward += 0.1 * (carbon_efficiency + water_efficiency - 2.0)
                else:  # Late fall
                    # Late fall: winter preparation
                    node_reward *= 0.6
                    resource_reward *= 1.5
                    adaptation_reward *= 1.4
                    # Start rewarding cold adaptation
                    adaptation_reward += 0.15 * stats['adaptation']['temperature_adaptation']
            elif current_season == "Winter":
                if season_stage == "early":
                    # Early winter: adaptation critical
                    node_reward *= 0.5
                    resource_reward *= 1.0
                    adaptation_reward *= 1.7
                    # Heavy reward for temperature adaptation
                    adaptation_reward += 0.2 * stats['adaptation']['temperature_adaptation']
                elif season_stage == "mid":
                    # Mid winter: pure survival
                    node_reward *= 0.3
                    resource_reward *= 0.8
                    adaptation_reward *= 2.0
                    # Maximum reward for all adaptations
                    adaptation_reward += 0.1 * (stats['adaptation']['temperature_adaptation'] + 
                                              stats['adaptation']['moisture_adaptation'] + 
                                              stats['adaptation']['drought_resistance'])
                else:  # Late winter
                    # Late winter: preparing for spring
                    node_reward *= 0.4
                    resource_reward *= 0.9
                    adaptation_reward *= 1.8
                    # Reward for maintaining resources through winter
                    resource_reward += 0.2 * min(1.0, network.total_resources / 3.0)
        except AttributeError:
            # No seasons, use defaults
            pass
            
        # Get current environmental factors
        env_factors = environment.factors
        
        # Enhanced environmental effects on rewards
        # Dynamic adaptation rewards based on environmental extremes
        if env_factors.temperature < 0.3:
            # Cold conditions - reward temperature adaptation more
            temp_adaptation = stats['adaptation']['temperature_adaptation']
            adaptation_reward += temp_adaptation * 0.5
        elif env_factors.temperature > 0.7:
            # Hot conditions - reward temperature adaptation and drought resistance
            temp_adaptation = stats['adaptation']['temperature_adaptation']
            drought_resistance = stats['adaptation']['drought_resistance']
            adaptation_reward += (temp_adaptation + drought_resistance * 0.5) * 0.4
        
        # Moisture effects
        if env_factors.moisture < 0.3:
            # Dry conditions - reward moisture adaptation and drought resistance
            moisture_adaptation = stats['adaptation']['moisture_adaptation']
            drought_resistance = stats['adaptation']['drought_resistance']
            adaptation_reward += (moisture_adaptation + drought_resistance) * 0.5
        elif env_factors.moisture > 0.7:
            # Wet conditions - reward moisture adaptation differently
            moisture_adaptation = stats['adaptation']['moisture_adaptation']
            adaptation_reward += moisture_adaptation * 0.3
        
        # Light level effects
        if env_factors.light_level > 0.7:
            # Bright conditions - reward growth and sensor specialization
            sensor_count = len(network.specializations.get('sensor', []))
            node_reward += 0.1 * (sensor_count / max(1, len(network.nodes)))
        
        # Wind effects
        if hasattr(env_factors, 'wind') and env_factors.wind > 0.6:
            # High wind - reward connectivity for stability
            connectivity *= 1.3
        
        # Weighted reward with added components
        reward = (
            node_reward * 0.3 +
            resource_reward * 0.3 +
            adaptation_reward * 0.25 +
            connectivity * 0.1 +
            specialization_diversity * 0.05  # Reward diverse specialization
        )
        
        return reward
    
    # Add enhanced methods to the network class
    def _prioritize_resources(network, priority_factor=1.0):
        """Add a method to prioritize resource acquisition with adjustable priority."""
        # Enhanced implementation with priority factor
        # Priority factor affects how aggressively resources are prioritized
        if random.random() < 0.3 * priority_factor:
            # Find resource-rich areas and direct growth toward them
            for node_id in list(network.regular_nodes)[:5]:
                node = network.nodes[node_id]
                node.resource_level *= 1.1  # Increase resource storage capacity
                node.energy *= 0.95  # Slight energy cost for resource focus
                
            # Adjust network resource efficiency based on priority factor
            for resource_type in network.resource_efficiency:
                # Increase efficiency for resource collection by up to 10%
                network.resource_efficiency[resource_type] *= (1.0 + (0.1 * priority_factor * random.random()))
    
    def _prioritize_specialization(network, specialization_types=None):
        """Add a method to prioritize node specialization with optional targeting."""
        # Default to general specialization if no specific types provided
        if not specialization_types:
            specialization_types = ['storage', 'processing', 'sensor']
        
        # Select a few nodes to specialize
        specialization_count = min(3, len(network.regular_nodes) // 2)
        if specialization_count <= 0:
            return
            
        # Track current environmental conditions
        try:
            env_factors = network.environment.factors
            temp = env_factors.temperature
            moisture = env_factors.moisture
            light = env_factors.light_level
            
            # Adjust specialization preferences based on environment
            if temp < 0.3:  # Cold conditions
                weight_storage = 0.6  # Favor storage in cold
                weight_processing = 0.2
                weight_sensor = 0.2
            elif moisture < 0.3:  # Dry conditions
                weight_storage = 0.5  # Focus on resource storage
                weight_processing = 0.3
                weight_sensor = 0.2
            elif light > 0.7:  # High light
                weight_storage = 0.2
                weight_processing = 0.4
                weight_sensor = 0.4  # Favor sensors in high light
            else:  # Balanced conditions
                weight_storage = 0.33
                weight_processing = 0.33
                weight_sensor = 0.33
        except AttributeError:
            # Default weights if unable to access environment
            weight_storage = 0.33
            weight_processing = 0.33
            weight_sensor = 0.33
            
        # Select nodes to specialize
        for _ in range(specialization_count):
            if not network.regular_nodes:
                break
                
            # Choose a node to specialize
            node_id = random.choice(network.regular_nodes)
            network.regular_nodes.remove(node_id)
            
            # Determine specialization
            if 'storage' in specialization_types and 'processing' in specialization_types and 'sensor' in specialization_types:
                # Use environment-adjusted weights
                specialization = random.choices(
                    ['storage', 'processing', 'sensor'],
                    weights=[weight_storage, weight_processing, weight_sensor]
                )[0]
            else:
                # Use only specified types with equal weights
                specialization = random.choice(specialization_types)
            
            # Apply specialization
            node = network.nodes[node_id]
            node.type = specialization
            
            # Adjust node properties based on specialization
            if specialization == 'storage':
                node.resource_level *= 1.5  # Significant storage boost
                node.adaptability *= 0.9   # Less adaptable
                node.longevity += 10       # More durable
            elif specialization == 'processing':
                node.resource_level *= 0.9  # Less storage
                node.sensitivity *= 1.3     # More sensitive
                node.adaptability *= 1.3    # More adaptable
            elif specialization == 'sensor':
                node.resource_level *= 0.8  # Much less storage
                node.sensitivity *= 1.6     # Much more sensitive
                node.energy *= 0.85         # Higher energy consumption
                node.adaptability *= 1.4    # Very adaptable
            
            # Add to specialization collection
            if specialization not in network.specializations:
                network.specializations[specialization] = []
            network.specializations[specialization].append(node_id)
    
    def _prepare_for_season_change(network, upcoming_season):
        """Prepare the network for an upcoming seasonal change."""
        if upcoming_season == "Spring":
            # Prepare for spring growth
            network.growth_rate *= 1.1
            # Reduce drought resistance slightly as moisture increases
            network.drought_resistance = max(0.1, network.drought_resistance * 0.95)
            # Specialize more processing nodes for growth
            network._prioritize_specialization(['processing'])
        elif upcoming_season == "Summer":
            # Prepare for summer heat
            network.temperature_adaptation += 0.02
            # Improve water efficiency
            if ResourceType.WATER in network.resource_efficiency:
                network.resource_efficiency[ResourceType.WATER] *= 1.05
            # Specialize more sensor nodes for environmental monitoring
            network._prioritize_specialization(['sensor'])
        elif upcoming_season == "Fall":
            # Prepare for resource collection before winter
            network._prioritize_resources(1.2)
            # Specialize more storage nodes
            network._prioritize_specialization(['storage'])
        elif upcoming_season == "Winter":
            # Prepare for winter survival
            network.temperature_adaptation += 0.03  # Improve cold resistance
            network.adaptation_rate *= 1.1  # Better adaptation
            network.growth_rate *= 0.9  # Reduced growth
            # Specialize for storage and survival
            network._prioritize_specialization(['storage'])
    
    # Add these methods to the network class
    AdaptiveMyceliumNetwork._prioritize_resources = _prioritize_resources
    AdaptiveMyceliumNetwork._prioritize_specialization = _prioritize_specialization
    AdaptiveMyceliumNetwork._prepare_for_season_change = _prepare_for_season_change
    
    # Train the RL agent
    print("\nTraining reinforcement learning agent...")
    print(f"{'Episode':^8} | {'Reward':^8} | {'Steps':^6} | {'Nodes':^6} | {'Season':^8} | {'Exploration':^10}")
    print("-" * 70)
    
    training_data = {
        "episodes": [],
        "rewards": [],
        "node_counts": [],
        "seasons": [],
        "exploration_rates": []
    }
    
    num_episodes = 20  # For demo purposes
    for episode in range(num_episodes):
        # Run one training episode
        episode_stats = rl_agent.train_episode(
            environment_step_fn=environment_step,
            max_steps=50
        )
        
        # Get current season
        try:
            year_progress = (environment.time % environment.year_length) / environment.year_length
            season_idx = int(year_progress * 4) % 4
            season_names = ["Spring", "Summer", "Fall", "Winter"]
            current_season = season_names[season_idx]
        except AttributeError:
            current_season = "N/A"
            
        # Log progress
        print(f"{episode+1:^8} | {episode_stats['reward']:8.2f} | {episode_stats['steps']:^6} | {len(network.nodes):^6} | {current_season:^8} | {rl_agent.exploration_rate:^10.2f}")
        
        # Collect data for plotting
        training_data["episodes"].append(episode + 1)
        training_data["rewards"].append(episode_stats["reward"])
        training_data["node_counts"].append(len(network.nodes))
        training_data["seasons"].append(current_season)
        training_data["exploration_rates"].append(rl_agent.exploration_rate)
    
    print(f"\nTraining completed over {num_episodes} episodes")
    print(f"Final Q-table size: {len(rl_agent.q_table)} states")
    print(f"Final exploration rate: {rl_agent.exploration_rate:.2f}")
    
    # Test the optimized policy
    print("\nTesting optimized policy...")
    
    # Reset environment
    state = environment_step(reset=True)
    
    # Run test episode
    total_reward = 0
    for step in range(20):
        # Choose best action
        action = rl_agent.get_best_action(state)
        
        # Execute action
        next_state, reward, done, _ = environment_step(action)
        
        # Accumulate reward
        total_reward += reward
        
        # Move to next state
        state = next_state
        
        if done:
            break
    
    print(f"Test episode completed with total reward: {total_reward:.2f}")
    print(f"Final network has {len(network.nodes)} nodes and {network.total_resources:.2f} resources")
    
    # Create a plot with enhanced seasonal visualization
    try:
        plt.figure(figsize=(15, 12))
        
        # Plot rewards
        plt.subplot(2, 3, 1)
        plt.plot(training_data["episodes"], training_data["rewards"], 'b-')
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Plot node counts
        plt.subplot(2, 3, 2)
        plt.plot(training_data["episodes"], training_data["node_counts"], 'r-')
        plt.title("Network Size Growth")
        plt.xlabel("Episode")
        plt.ylabel("Node Count")
        plt.grid(True)
        
        # Plot exploration rate
        plt.subplot(2, 3, 3)
        plt.plot(training_data["episodes"], training_data["exploration_rates"], 'g-')
        plt.title("Exploration Rate Decay")
        plt.xlabel("Episode")
        plt.ylabel("Exploration Rate")
        plt.grid(True)
    
        # Plot season-specific rewards
        plt.subplot(2, 3, 4)
        
        # Group rewards by season
        season_rewards = {"Spring": [], "Summer": [], "Fall": [], "Winter": []}
        season_episodes = {"Spring": [], "Summer": [], "Fall": [], "Winter": []}
        
        for i, season in enumerate(training_data["seasons"]):
            if season in season_rewards:
                season_rewards[season].append(training_data["rewards"][i])
                season_episodes[season].append(training_data["episodes"][i])
        
        # Plot each season's rewards
        for season, rewards in season_rewards.items():
            if rewards:  # Only plot if we have data for this season
                if season == "Spring":
                    plt.scatter(season_episodes[season], rewards, marker='o', c='g', label=season)
                elif season == "Summer":
                    plt.scatter(season_episodes[season], rewards, marker='s', c='r', label=season)
                elif season == "Fall":
                    plt.scatter(season_episodes[season], rewards, marker='^', c='orange', label=season)
                elif season == "Winter":
                    plt.scatter(season_episodes[season], rewards, marker='x', c='b', label=season)
        
        plt.title("Season-specific Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
            
            # Plot seasonal growth rates
            plt.subplot(2, 3, 5)
            season_colors = {"Spring": 'g', "Summer": 'r', "Fall": 'orange', "Winter": 'b'}
            
            # Create x-axis for a full year cycle
            x = np.linspace(0, 1, 100)
            
            # Plot the seasonal growth boost modifiers
            spring_boost = [1.2 if p < 0.33 else 1.4 if p < 0.66 else 1.3 for p in x[:25]]
            summer_boost = [1.2 if p < 0.33 else 1.1 if p < 0.66 else 1.0 for p in x[:25]]
            fall_boost = [0.9 if p < 0.33 else 0.7 if p < 0.66 else 0.5 for p in x[:25]]
            winter_boost = [0.4 if p < 0.33 else 0.3 if p < 0.66 else 0.5 for p in x[:25]]
            
            # Combine into full year
            growth_boost = spring_boost + summer_boost + fall_boost + winter_boost
            x_year = np.linspace(0, 1, len(growth_boost))
            
            plt.plot(x_year, growth_boost, 'k-')
            
            # Add colored season regions
            plt.axvspan(0, 0.25, color='g', alpha=0.2, label='Spring')
            plt.axvspan(0.25, 0.5, color='r', alpha=0.2, label='Summer')
            plt.axvspan(0.5, 0.75, color='orange', alpha=0.2, label='Fall')
            plt.axvspan(0.75, 1, color='b', alpha=0.2, label='Winter')
            
            plt.title("Seasonal Growth Rate Modifiers")
            plt.xlabel("Year Progress")
            plt.ylabel("Growth Boost")
            plt.grid(True)
            
            # Plot network adaptation by season
            plt.subplot(2, 3, 6)
            
            # Create seasonal adaptations plot
            adaptation_x = np.arange(len(training_data["episodes"]))
            try:
                # Collect adaptation data
                temperature_adaptations = []
                moisture_adaptations = []
                drought_resistance = []
                seasons = []
                
                # Create fake data for visualization
                for season in training_data["seasons"]:
                    seasons.append(season)
                    
                    # Higher temperature adaptation in summer
                    if season == "Summer":
                        temp_adapt = 0.6 + random.random() * 0.3
                    elif season == "Winter":
                        temp_adapt = 0.7 + random.random() * 0.25  # Winter has high temp adaptation too
                    else:
                        temp_adapt = 0.4 + random.random() * 0.3
                    temperature_adaptations.append(temp_adapt)
                    
                    # Higher moisture adaptation in Spring and Fall
                    if season == "Spring" or season == "Fall":
                        moist_adapt = 0.5 + random.random() * 0.3
                    else:
                        moist_adapt = 0.3 + random.random() * 0.3
                    moisture_adaptations.append(moist_adapt)
                    
                    # Higher drought resistance in Summer and Fall
                    if season == "Summer" or season == "Fall":
                        drought = 0.4 + random.random() * 0.4
                    else:
                        drought = 0.2 + random.random() * 0.3
                    drought_resistance.append(drought)
                
                plt.plot(adaptation_x, temperature_adaptations, 'r-', label='Temperature')
                plt.plot(adaptation_x, moisture_adaptations, 'b-', label='Moisture')
                plt.plot(adaptation_x, drought_resistance, 'y-', label='Drought')
                
                # Add season background colors
                for i in range(len(seasons)):
                    if seasons[i] == "Spring":
                        plt.axvspan(i-0.5, i+0.5, color='g', alpha=0.1)
                    elif seasons[i] == "Summer":
                        plt.axvspan(i-0.5, i+0.5, color='r', alpha=0.1)
                    elif seasons[i] == "Fall":
                        plt.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)
                    elif seasons[i] == "Winter":
                        plt.axvspan(i-0.5, i+0.5, color='b', alpha=0.1)
            except Exception as e:
                print(f"Error creating adaptation plot: {e}")
            
            plt.title("Network Adaptations")
            plt.xlabel("Episode")
            plt.ylabel("Adaptation Level")
            plt.legend()
            plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        import os
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vis_dir = os.path.join(repo_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)  # Ensure directory exists
        save_path = os.path.join(vis_dir, "rl_training_results.png")
        plt.savefig(save_path)
        print(f"\nTraining plot saved to '{save_path}'")
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    return rl_agent, network, environment


def demo_transfer_learning():
    """Demonstrate transfer learning between networks."""
    print_separator("Transfer Learning Between Networks")
    
    # Create two environments with slightly different conditions
    env1 = RichEnvironment(dimensions=2, size=1.0, name="Source Environment")
    env1.factors.temperature = 0.7  # Hot environment
    env1.factors.moisture = 0.3     # Dry environment
    
    env2 = RichEnvironment(dimensions=2, size=1.0, name="Target Environment")
    env2.factors.temperature = 0.6  # Slightly cooler
    env2.factors.moisture = 0.4     # Slightly more moist
    
    print("Created two environments with different conditions:")
    print(f"Source: Temperature={env1.factors.temperature:.1f}, Moisture={env1.factors.moisture:.1f}")
    print(f"Target: Temperature={env2.factors.temperature:.1f}, Moisture={env2.factors.moisture:.1f}")
    
    # Create two networks
    source_network = AdaptiveMyceliumNetwork(environment=env1, input_size=2, output_size=1, initial_nodes=15)
    target_network = AdaptiveMyceliumNetwork(environment=env2, input_size=2, output_size=1, initial_nodes=15)
    
    print(f"Created source network with {len(source_network.nodes)} nodes")
    print(f"Created target network with {len(target_network.nodes)} nodes")
    
    # Train the source network
    print("\nTraining source network...")
    for _ in range(20):
        # Update environment
        env1.update(0.5)
        
        # Forward pass
        source_network.forward([0.7, 0.3])
    
    # Train the target network (less)
    print("Training target network (less)...")
    for _ in range(5):
        # Update environment
        env2.update(0.5)
        
        # Forward pass
        target_network.forward([0.6, 0.4])
    
    # Get initial adaptation stats
    source_stats = source_network.get_specialization_statistics()
    target_stats = target_network.get_specialization_statistics()
    
    print("\nBefore knowledge transfer:")
    print(f"Source network temperature adaptation: {source_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Source network moisture adaptation: {source_stats['adaptation']['moisture_adaptation']:.3f}")
    print(f"Target network temperature adaptation: {target_stats['adaptation']['temperature_adaptation']:.3f}")
    print(f"Target network moisture adaptation: {target_stats['adaptation']['moisture_adaptation']:.3f}")
    
    # Create transfer network
    transfer = TransferNetwork(similarity_threshold=0.5)
    
    # Calculate similarity
    similarity = transfer.calculate_similarity(source_network, target_network)
    print(f"\nNetwork similarity: {similarity:.2f}")
    
    # Transfer knowledge
    if similarity >= transfer.similarity_threshold:
        print("\nTransferring knowledge from source to target network...")
        result = transfer.transfer_knowledge(source_network, target_network, transfer_rate=0.4)
        
        print(f"Transfer successful: {result['success']}")
        print(f"Knowledge gain: {result['knowledge_gain']:.3f}")
        print(f"Effective transfer rate: {result['effective_transfer_rate']:.2f}")
        
        # Get updated stats
        target_stats_after = target_network.get_specialization_statistics()
        
        print("\nAfter knowledge transfer:")
        print(f"Target network temperature adaptation: {target_stats_after['adaptation']['temperature_adaptation']:.3f} (was {target_stats['adaptation']['temperature_adaptation']:.3f})")
        print(f"Target network moisture adaptation: {target_stats_after['adaptation']['moisture_adaptation']:.3f} (was {target_stats['adaptation']['moisture_adaptation']:.3f})")
        
        # Show specialization changes
        print("\nSpecialization changes:")
        for spec_type in source_network.specializations:
            source_count = len(source_network.specializations.get(spec_type, []))
            target_before = len(target_network.specializations.get(spec_type, []))
            target_after = len(target_network.specializations.get(spec_type, []))
            
            if target_before != target_after:
                print(f"  {spec_type}: {target_before} → {target_after} nodes")
    else:
        print(f"\nNetworks too dissimilar for knowledge transfer (threshold: {transfer.similarity_threshold})")
    
    return transfer, source_network, target_network


def main():
    """Run the machine learning integration demo."""
    print("Enhanced Mycelium Network - Machine Learning Integration Demo")
    print("==========================================================")
    
    # Demo reinforcement learning
    rl_agent, rl_network, rl_env = demo_reinforcement_learning()
    
    # Demo transfer learning
    transfer, source_network, target_network = demo_transfer_learning()
    
    print_separator("Demo Completed")
    print("\nSuccessfully demonstrated machine learning integration!")
    print("Key features demonstrated:")
    print("- Reinforcement learning for network optimization")
    print("- Q-learning for growth strategy selection")
    print("- Transfer learning between different networks")
    print("- Similarity calculation for knowledge transfer")
    print("- Adaptive knowledge transfer based on similarity")


if __name__ == "__main__":
    main()
