"""
Reinforcement learning integration for the mycelium network.

This module provides a reinforcement learning agent that can optimize
network growth and adaptation in response to environmental stimuli.
"""

import random
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any, Callable


class ReinforcementLearner:
    """
    Reinforcement learning agent for mycelium network optimization.
    
    This agent uses Q-learning to optimize growth strategies and
    resource allocation in response to environmental conditions.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        memory_size: int = 2000
    ):
        """
        Initialize the reinforcement learning agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
            memory_size: Size of experience replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize Q-table
        self.q_table = {}
        
        # Learning statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0,
            "avg_reward": 0,
            "min_reward": float('inf'),
            "max_reward": float('-inf'),
            "exploration_rate": self.exploration_rate
        }
    
    def _get_state_key(self, state):
        """Convert state to hashable key for Q-table."""
        if isinstance(state, np.ndarray):
            return tuple(state.tolist())
        elif isinstance(state, list):
            return tuple(state)
        return state
    
    def _ensure_state_exists(self, state_key):
        """Ensure state exists in Q-table."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
    
    def choose_action(self, state):
        """
        Choose an action based on current state using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        state_key = self._get_state_key(state)
        self._ensure_state_exists(state_key)
        
        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: choose best action
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """
        Train agent by replaying experiences.
        
        Args:
            batch_size: Number of experiences to replay
            
        Returns:
            Average loss across batch
        """
        if len(self.memory) < batch_size:
            return 0
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            self._ensure_state_exists(state_key)
            self._ensure_state_exists(next_state_key)
            
            # Current Q-value
            current_q = self.q_table[state_key][action]
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
            
            # Update Q-value
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
            
            # Calculate loss
            loss = (target_q - current_q) ** 2
            total_loss += loss
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
        
        # Update training stats
        self.training_stats["exploration_rate"] = self.exploration_rate
        
        return total_loss / batch_size
    
    def train_episode(
        self,
        environment_step_fn: Callable,
        max_steps: int = 200,
        render: bool = False
    ):
        """
        Train the agent for one episode.
        
        Args:
            environment_step_fn: Function to step the environment
            max_steps: Maximum episode steps
            render: Whether to render the environment
            
        Returns:
            Episode statistics
        """
        # Reset environment and get initial state
        state = environment_step_fn(action=None, reset=True)
        
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            # Choose action
            action = self.choose_action(state)
            
            # Take action
            next_state, reward, done, info = environment_step_fn(action=action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Train on past experiences
            loss = self.replay()
            
            # Optional rendering
            if render and hasattr(environment_step_fn, "render"):
                environment_step_fn.render()
            
            # Check if done
            if done:
                break
        
        # Update episode statistics
        self.training_stats["episodes"] += 1
        self.training_stats["total_reward"] += total_reward
        self.training_stats["avg_reward"] = self.training_stats["total_reward"] / self.training_stats["episodes"]
        self.training_stats["min_reward"] = min(self.training_stats["min_reward"], total_reward)
        self.training_stats["max_reward"] = max(self.training_stats["max_reward"], total_reward)
        
        # Return episode statistics
        return {
            "episode": self.training_stats["episodes"],
            "reward": total_reward,
            "steps": step + 1,
            "exploration_rate": self.exploration_rate,
            "q_table_size": len(self.q_table)
        }
    
    def get_best_action(self, state):
        """
        Get the best action for a state without exploration.
        
        Args:
            state: Current state
            
        Returns:
            Best action index
        """
        state_key = self._get_state_key(state)
        self._ensure_state_exists(state_key)
        
        return np.argmax(self.q_table[state_key])
    
    def save_model(self, filename):
        """
        Save the Q-table to a file.
        
        Args:
            filename: File to save to
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'training_stats': self.training_stats
            }, f)
    
    def load_model(self, filename):
        """
        Load the Q-table from a file.
        
        Args:
            filename: File to load from
            
        Returns:
            True if loaded successfully
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.state_size = data['state_size']
                self.action_size = data['action_size']
                self.training_stats = data['training_stats']
            return True
        except:
            return False
