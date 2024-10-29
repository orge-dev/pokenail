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

    def start(self):
        """Start the emulator and run the main loop."""
        try:
            while True:
                if not self.pyboy.tick():  # Update the emulator state
                    break  # Exit if the emulator signals to stop
        except KeyboardInterrupt:
            logging.info("Emulation interrupted by user.")
        except Exception as e:
            logging.error(f"Error during update: {e}")
        finally:
            self.close()

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
        # Define the mapping of actions to button presses
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

        # Perform the action based on the input
        if action in action_map:
            func_action = action_map[action]
            func_action()
            logging.info(f"Performed action: {action}")
        else:
            logging.warning(f"Unknown action: {action}")
