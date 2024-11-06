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
from replay_buffer import ReplayBuffer


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
    def __init__(self, learning_rate=0.05, discount_factor=0.9, headless=False):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED, headless=headless)
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.reset()

    def reset(self):
        self.controller.load_state()

        # Maybe group these as a subclass
        self.replay_buffer = ReplayBuffer()
        self.visited_coords = set()
        self.battle = self.controller.is_in_battle()
        self.battle_reward_applied = False
        self.current_step = 0
        self.total_reward = 0
        self.steps_to_battle = None
        self.last_distance_reward = None
        self.position = self.controller.get_global_coords()

        initial_state = {
            "position": self.position,
            "battle": self.battle,
        }
        self.previous_state = initial_state
        return initial_state

    def calculate_distance_metrics(self, position):
        target_position = (309, 99)
        distance = np.sqrt(
            (position[0] - target_position[0]) ** 2
            + (position[1] - target_position[1]) ** 2
        )
        if distance > 20:
            distance_reward = 0.0
        else:
            distance_reward = 1000.0 / (distance + 1)
        return distance, distance_reward

    def calculate_reward(self, position):
        position_tuple = tuple(position)

        # Distance-based reward
        current_distance, distance_reward = self.calculate_distance_metrics(
            position_tuple
        )

        # Print first distance reward or significant changes
        if self.last_distance_reward is None:
            print(f"\nInitial distance reward: {distance_reward:.2f}")
            self.last_distance_reward = distance_reward
        else:
            change = abs(distance_reward - self.last_distance_reward)
            if change >= 5.0:
                print(
                    f"\nSignificant distance change! Old: {self.last_distance_reward:.2f}, New: {distance_reward:.2f}"
                )
                self.last_distance_reward = distance_reward

        # Battle reward (scaled by steps)
        if not self.battle_reward_applied and self.battle:
            battle_reward = 100000 * (1 / self.current_step)
            self.battle_reward_applied = True
            print(f"\nBattle found! Position: {position}, Battle: {self.battle}")
            print(f"Battle reward: {battle_reward}")
            # NOTE: only battle reward is active right now
            # Others are still calculated above but not used.
            # And done = self.battle below, so the episode ends on finding a battle
            return battle_reward

        return 0  # No reward for non-battle steps

    def step(self, action=None, manual=False, agent=None):
        self.current_step += 1
        if not manual and action is not None:
            self.controller.perform_action(action)

        self.controller.pyboy.tick()
        self.battle = self.controller.is_in_battle()
        self.position = self.controller.get_global_coords()
        self.visited_coords.add(tuple(self.position))  # Add this line

        if self.battle and self.steps_to_battle is None:
            self.steps_to_battle = self.current_step

        position = self.controller.get_global_coords()
        step_reward = self.calculate_reward(position)
        self.total_reward += step_reward

        next_state = {
            "position": position,
            "battle": self.battle,
        }

        if not manual and agent is not None:
            agent.update_q_table(self.previous_state, action, next_state, step_reward)

        done = self.battle and not manual  # dont end on battle if manual

        experience = {
            "state": self.previous_state,
            "action": action,
            "reward": step_reward,
            "next_state": next_state,
            "done": done,
        }

        self.replay_buffer.add(experience)

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
            "total_steps": self.current_step,
            "steps_to_battle": self.steps_to_battle,
            "total_reward": self.total_reward,
            "final_position": self.position,
            "visited_coords": list(
                self.visited_coords
            ),  # Convert set to list for serialization
            "battle": self.battle,
            "battle_reward_applied": self.battle_reward_applied,
            "last_distance_reward": self.last_distance_reward,
        }
        os.makedirs("episodes", exist_ok=True)
        with open(f"episodes/episode_{episode_id}.pkl", "wb") as f:
            pickle.dump(stats, f)

        os.makedirs("replays", exist_ok=True)
        self.replay_buffer.save(f"replays/replay_{episode_id}.pkl")

    def close(self):
        self.controller.close()
