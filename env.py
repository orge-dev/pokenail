import os
import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from actions import (
    Actions,
)  # Replace `actions` with the actual module name if different
from config import EMULATION_SPEED, ROM_PATH
from game_controller import GameController
from replay_buffer import ReplayBuffer


# This is the information we save to replays and train from
@dataclass(frozen=True)
class EnvironmentState:
    position: tuple
    battle: bool
    # prev_position: tuple | None
    has_oaks_parcel: bool
    has_pokedex: bool
    menu_y: int
    menu_x: int
    menu_selected: int


class EnvRed:
    def __init__(self, headless=False):
        self.controller = GameController(ROM_PATH, EMULATION_SPEED, headless=headless)

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
        # self.prev_position = None
        self.local_position = self.controller.get_game_coords()
        self.position = self.controller.get_global_coords()
        self.previous_items = dict()
        self.nearly_visited_coords = set()

        # Make sure to do this on reset so that we don't get "null" inputs on step 1
        # giving reward even though we haven't actually moved
        self.update_nearly_visited_coords()

        # Debug vars
        self.last_cumulative_reward = None

        self.previous_state = self.state_for_agent()

        
    def state_for_agent(self):
        return EnvironmentState(
            position=self.position,
            battle=self.battle,
            # prev_position=self.prev_position,
            has_oaks_parcel=self.has_oaks_parcel(),
            has_pokedex=self.has_pokedex(),
            menu_y=self.controller.mem(self.controller.MEMORY_MENU_Y),
            menu_x=self.controller.mem(self.controller.MEMORY_MENU_X),
            menu_selected=self.controller.mem(self.controller.MEMORY_MENU_SELECTED),
        )

    def has_oaks_parcel(self):
        return "Oak's Parcel" in self.controller.get_items()

    def has_pokedex(self):
        # TODO: Probably use MissableObject from pokemonred_puffer or something similar
        pokedex_sprite_byte = self.controller.mem(0xD5A6 + 5)
        pokedex_sprite_bit = pokedex_sprite_byte >> 7
        return bool(pokedex_sprite_bit)

    def calculate_distance_reward(self, position):
        target_position = (309, 99)
        distance = np.sqrt(
            (position[0] - target_position[0]) ** 2
            + (position[1] - target_position[1]) ** 2
        )
        if distance > 20:
            distance_reward = 0.0
        else:
            distance_reward = 1000.0 / (distance + 1)
        return distance_reward

    def calculate_reward(self, position):
        position_tuple = tuple(position)

        # Distance-based reward
        distance_reward = self.calculate_distance_reward(position_tuple)

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
        # if not self.battle_reward_applied and self.battle:
        #     battle_reward = 100000 * (1 / self.current_step)
        #     self.battle_reward_applied = True
        #     print(f"\nBattle found! Position: {position}, Battle: {self.battle}")
        #     print(f"Battle reward: {battle_reward}")
        #     # NOTE: only battle reward is active right now
        #     # Others are still calculated above but not used.
        #     # And done = self.battle below, so the episode ends on finding a battle
        #     return battle_reward

        # return 0  # no reward for non-battle steps

        return len(self.nearly_visited_coords) - self.previous_nearly_visited_coords

    def update_nearly_visited_coords(self):
        # Add 5x5 area around current position to nearly_visited_coords
        for dy in range(-2, 3):  # -2 to 2 for 5x5 area
            for dx in range(-2, 3):
                nearby_pos = (self.position[0] + dy, self.position[1] + dx)
                self.nearly_visited_coords.add(nearby_pos)

    def step(self, action=None, manual=False, agent=None):
        self.current_step += 1
        if not manual and action is not None:
            self.controller.perform_action(action)

        self.controller.pyboy.tick()
        self.battle = self.controller.is_in_battle()
        self.local_position = self.controller.get_game_coords()
        #self.prev_position = self.position
        self.position = self.controller.get_global_coords()
        self.visited_coords.add(tuple(self.position))  # Add this line

        self.previous_nearly_visited_coords = len(self.nearly_visited_coords)
        self.update_nearly_visited_coords()

        # Check for new/changed items
        current_items = self.controller.get_items()
        for item_name, quantity in current_items.items():
            prev_quantity = self.previous_items.get(item_name, 0)
            if quantity != prev_quantity:
                gained = quantity - prev_quantity
                print(f"\nGained {gained}x {item_name}!")
        self.previous_items = current_items

        if self.battle and self.steps_to_battle is None:
            self.steps_to_battle = self.current_step

        step_reward = self.calculate_reward(self.position)
        self.total_reward += step_reward

        next_state = self.state_for_agent()

        # # Print changed fields
        # for field in next_state.__dataclass_fields__:
        #     old_val = getattr(self.previous_state, field)
        #     new_val = getattr(next_state, field)
        #     # if old_val != new_val:
        #     #     print(f"{field} changed: {old_val} -> {new_val}")

        if not manual and agent is not None:
            agent.update_q_table(self.previous_state, action, next_state, step_reward)

        # done = self.battle and not manual  # dont end on battle if manual
        done = False

        cumulative_reward = len(self.nearly_visited_coords)
        if manual:
            if cumulative_reward != self.last_cumulative_reward:
                print("cumulative reward", cumulative_reward)
        self.last_cumulative_reward = cumulative_reward

        # print("seen coords", cumulative_reward)

        experience = {
            "state": self.previous_state,
            "action": action,
            "reward": step_reward,
            "cumulative_reward": cumulative_reward,
            "next_state": next_state,
            "done": done,
        }

        self.replay_buffer.add(experience)

        self.previous_state = next_state
        return next_state, step_reward, cumulative_reward, done, {}

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

    def save_state(self, save_dir):
        raise NotImplementedError()

    def close(self):
        self.controller.close()
