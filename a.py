# game_controller.py

import logging
from pyboy import PyBoy
from config import ROM_PATH, EMULATION_SPEED  # Make sure these are defined in config

logging.basicConfig(level=logging.INFO)

class GameController:
    def __init__(self, rom_path, emulation_speed=1.0):
        """
        Initialize the GameController with a ROM path and emulation speed.
        
        Args:
            rom_path (str): Path to the ROM file.
            emulation_speed (float): Speed factor (1.0 = normal, >1.0 = faster, <1.0 = slower)
        """
        self.pyboy = PyBoy(rom_path)
        if not self.pyboy:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM.")
        
        self.pyboy.set_emulation_speed(emulation_speed)  # Set the emulation speed

    def step(self):
        """Step the emulator."""
        return self.pyboy.tick()  # Returns whether the emulator is still running

    def save_state(self, state_filename="game_state.save"):
        with open(state_filename, "wb") as f:
            self.pyboy.save_state(f)
        logging.info(f"Game state saved to {state_filename}")

    def load_state(self, state_filename="start.save"):
        with open(state_filename, "rb") as f:
            self.pyboy.load_state(f)
        logging.info(f"Game state loaded from {state_filename}")

    def close(self):
        self.pyboy.stop()
        logging.info("Emulator stopped.")

    def update(self):
        """Update the game state; implement your logic here."""
        # Implement logic to retrieve and return game state
        return {}

    def perform_action(self, action):
        """Perform the action given by the AI agent."""
        print(f'the action is {action}')
        action_map = {
            "A": lambda: self.pyboy.button_press('a'),
            "B": lambda: self.pyboy.button_press('b'),
            "START": lambda: self.pyboy.button_press('start'),
            "SELECT": lambda: self.pyboy.button_press('select'),
            "UP": lambda: self.pyboy.button_press('up'),
            "DOWN": lambda: self.pyboy.button_press('down'),
            "LEFT": lambda: self.pyboy.button_press('left'),
            "RIGHT": lambda: self.pyboy.button_press('right'),
        }

        action_map_release = {
            "A": lambda: self.pyboy.button_release('a'),
            "B": lambda: self.pyboy.button_release('b'),
            "START": lambda: self.pyboy.button_release('start'),
            "SELECT": lambda: self.pyboy.button_release('select'),
            "UP": lambda: self.pyboy.button_release('up'),
            "DOWN": lambda: self.pyboy.button_release('down'),
            "LEFT": lambda: self.pyboy.button_release('left'),
            "RIGHT": lambda: self.pyboy.button_release('right'),
        }

        button_hold_ticks = 20
        button_release_ticks = 2

        if action in action_map:
            func_action = action_map[action]
            func_action()

            self.pyboy.tick(button_hold_ticks)

            end_func_action = action_map_release[action]
            end_func_action()
            self.pyboy.tick(button_release_ticks)

            logging.info(f"Performed action: {action}")
        else:
            logging.warning(f"Unknown action: {action}")

    def run(self, agent):           
        try:
            agent.run(self, checkpoint="q_table_final.pkl", max_steps=100)
        finally:
            self.close()

# main.py

import logging
from game_controller import GameController
from ai_agent import AIAgent
from config import ROM_PATH, EMULATION_SPEED  # Ensure these are correctly defined in config

def main():
    game_controller = GameController(rom_path=ROM_PATH, emulation_speed=EMULATION_SPEED)
    agent = AIAgent()

    try:
        game_controller.run(agent)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        game_controller.close()

if __name__ == "__main__":
    main()

# ai_agent.py

import numpy as np
import random
import pickle
from collections import defaultdict
from actions import Actions  # Ensure Actions enum is imported

class AIAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        num_actions = len(Actions.list())
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def run(self, game_controller, checkpoint="q_table_final.pkl", max_steps=100):
        self.load_q_table(checkpoint)

        state = "initial_state"  # Define a proper state based on your game_controller update
        done = False
        step = 0

        while not done and step < max_steps:
            action = self.select_action(state)
            game_controller.perform_action(action)
            next_state = game_controller.update()
            reward = 1  # Update reward logic based on game status
            done = False  # Set this properly
            state = next_state
            step += 1
        print("Run completed.")

    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(Actions.list())
        else:
            action_index = np.argmax(self.q_table[state])
            return Actions.list()[action_index]

    def update(self, state, action, reward, next_state):
        action_index = Actions.list().index(action)
        best_next_action_value = np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.learning_rate * (
            reward + self.discount_factor * best_next_action_value - self.q_table[state][action_index]
        )
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as file:
            self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())), pickle.load(file))
