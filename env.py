from abc import ABC, abstractmethod
from game_controller import GameController
from config import ROM_PATH, EMULATION_SPEED
from global_map import local_to_global
import numpy as np
from collections import defaultdict
from actions import Actions  # Replace `actions` with the actual module name if different


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
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.visited_coords = set()  # Track visited coordinates
   
    def reset(self):
        self.controller.load_state()
        position = self.controller.get_global_coords()
        initial_state = {
            "position": position,
            "score": 0,
        }
        self.previous_state = initial_state  # Initialize previous_state
        return initial_state


    def step(self, action=None, manual=False):
        """Execute a step in the environment, optionally with manual control."""
        if not manual and action is not None:
            # Perform the action only if not in manual mode and an action is provided
            self.controller.perform_action(action)
            
        self.controller.pyboy.tick()  # Advance the emulator state

        position = self.controller.get_global_coords()
        next_state = {
            "position": position,
            "score": 5,
            "exploration_reward": self.calculate_exploration_reward(position),
        }
        reward = 1  # Replace with actual reward calculation
        done = False  # Set to True if the episode ends

        # Update the Q-table only in automated mode
        if not manual:
            self.update_q_table(self.previous_state, action, next_state, reward)
        
        self.previous_state = next_state  # Update previous state

        return next_state, reward, done, {}
    
    def calculate_exploration_reward(self, position):
        """Calculate an exploration reward for visiting new positions."""
        position_tuple = tuple(position)
        if position_tuple not in self.visited_coords:
            self.visited_coords.add(position_tuple)  # Mark position as visited
            return 10  # Exploration reward for new positions
        return 0  # No reward if position has been visited before

    def update_q_table(self, state, action, next_state, reward):
        """Updates the Q-table for the current environment state."""
        position = next_state['position']
        exploration_reward = self.calculate_exploration_reward(position)
        total_reward = reward + exploration_reward

        # Update Q-table with the new reward
        state, next_state = tuple(state.items()), tuple(next_state.items())
        action_index = Actions.list().index(action)
        best_next_action_value = np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.learning_rate * (
            total_reward
            + self.discount_factor * best_next_action_value
            - self.q_table[state][action_index]
        )

    def close(self):
        self.controller.close()
