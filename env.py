from abc import ABC, abstractmethod
from game_controller import GameController
from config import ROM_PATH, EMULATION_SPEED  # Ensure these are defined in config

from global_map import local_to_global


class AbstractEnvironment(ABC):
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Take an action in the environment and return the next state, reward, done, and additional info."""
        pass

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255


class env_red(AbstractEnvironment):
    def __init__(self):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)

    def reset(self):
        self.controller.load_state()
        # Initialize the game state, such as player position and score
        position = self.controller.get_global_coords()
        initial_state = {
            "position": position,
            "score": 0,
        }  # Replace with actual state logic
        return initial_state

    def step(self):
        self.controller.pyboy.tick()

        # Update the state after action
        position = self.controller.get_global_coords()
        next_state = {"position": position, "score": 5}  # Replace with real state info
        reward = 1  # Define reward logic
        done = False  # Set True if episode ends
        return next_state, reward, done, {}

    def close(self):
        self.controller.close()
