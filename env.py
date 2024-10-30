# your_environment.py

from abc import ABC, abstractmethod

class AbstractEnvironment(ABC):
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Take an action in the environment and return the next state, reward, done, and additional info."""
        pass

class env_red(AbstractEnvironment):
    def reset(self):
        # Initialize and return the starting state
        return "initial_state"

    def step(self, action):
        # Execute the action and return the next state, reward, done flag, and additional info
        next_state = "next_state"  # Update based on your logic
        reward = 1  # Replace with your reward logic
        done = False  # Set to True when the episode ends
        return next_state, reward, done, {}
