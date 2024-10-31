
from abc import ABC, abstractmethod
from game_controller import GameController
from config import ROM_PATH, EMULATION_SPEED  # Ensure these are defined in config

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
    def __init__(self):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)

    def reset(self):
        self.controller.load_state()
        # Initialize the game state, such as player position and score
        initial_state = {"position": (0, 0), "score": 0}  # Replace with actual state logic
        return initial_state

    def step(self, action):
        self.controller.perform_action(action)
        self.controller.pyboy.tick()

        # Update the state after action
        next_state = {"position": (1, 2), "score": 5}  # Replace with real state info
        reward = 1  # Define reward logic
        done = False  # Set True if episode ends
        return next_state, reward, done, {}

    def close(self):
        self.controller.close()