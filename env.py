from abc import ABC, abstractmethod
import os
import pickle
from game_controller import GameController
from config import ROM_PATH, EMULATION_SPEED
from global_map import local_to_global
import numpy as np
from collections import defaultdict
from actions import (
    Actions,
)  # Replace `actions` with the actual module name if different


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


class env_red(AbstractEnvironment):
    def __init__(self, learning_rate=0.05, discount_factor=0.9):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.reset()

    def reset(self):
        self.controller.load_state()

        self.visited_coords = set()  # Track visited coordinates
        self.battle = self.controller.is_in_battle()
        self.battle_reward_applied = False
        self.current_step = 0
        self.total_reward = 0  # Track total reward for episode
        self.steps_to_battle = None

        position = self.controller.get_global_coords()
        initial_state = {
            "position": position,
            "battle": self.battle,
        }
        self.previous_state = initial_state
        return initial_state

    def calculate_reward(self, position):
        reward = 0
        position_tuple = tuple(position)

        # Track exploration and battle rewards separately
        exploration_reward = 0
        battle_reward = 0

        # Exploration reward
        if position_tuple not in self.visited_coords:
            exploration_reward = 5
            reward += exploration_reward
            self.visited_coords.add(position_tuple)
            print(f"\nNew area explored! Position: {position}, Battle: {self.battle}")
            print(f"Exploration reward: {exploration_reward}")
        else:
            exploration_reward = -0.5
            reward += exploration_reward

        # Battle reward
        if not self.battle_reward_applied and self.battle:
            steps_taken = self.current_step
            battle_reward = 10000 * (1.0 / steps_taken)
            reward += battle_reward
            self.battle_reward_applied = True
            print(f"\nBattle found! Position: {position}, Battle: {self.battle}")
            print(f"Battle reward: {battle_reward}")

        return reward

    def step(self, action=None, manual=False):
        self.current_step += 1

        if not manual and action is not None:
            self.controller.perform_action(action)

        self.controller.pyboy.tick()

        if self.controller.is_in_battle():
            self.battle = True
            if self.steps_to_battle is None:
                self.steps_to_battle = self.current_step
        else:
            self.battle = False

        position = self.controller.get_global_coords()
        step_reward = self.calculate_reward(position)
        self.total_reward += step_reward

        next_state = {
            "position": position,
            "battle": self.battle,
        }

        done = False

        if not manual:
            self.update_q_table(self.previous_state, action, next_state, step_reward)

        self.previous_state = next_state

        return next_state, step_reward, done, {}

    def update_q_table(self, state, action, next_state, reward):
        """Updates the Q-table for the current environment state and reward."""
        state, next_state = tuple(state.items()), tuple(next_state.items())
        action_index = Actions.list().index(action)
        best_next_action_value = np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_action_value
            - self.q_table[state][action_index]
        )

    def save_episode_stats(self, episode_id):
        """Save episode statistics."""
        stats = {
            "visited_coords": list(self.visited_coords),
            "steps_to_battle": self.steps_to_battle,
            "total_steps": self.current_step,
            "total_reward": self.total_reward,  # Add this line
        }
        os.makedirs("episodes", exist_ok=True)
        with open(f"episodes/episode_{episode_id}.pkl", "wb") as f:
            pickle.dump(stats, f)

    def close(self):
        self.controller.close()
