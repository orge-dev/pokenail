import logging
from pyboy import PyBoy
from config import ROM_PATH, EMULATION_SPEED  # Ensure these are defined in config

logging.basicConfig(level=logging.INFO)

class GameController:
    def __init__(self, rom_path, emulation_speed=1.0):
        self.pyboy = PyBoy(rom_path)
        if not self.pyboy:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM.")
        
        self.pyboy.set_emulation_speed(emulation_speed)
        logging.info("GameController initialized with ROM: %s", rom_path)

    def save_state(self, state_filename="game_state.save"):
        with open(state_filename, "wb") as f:
            self.pyboy.save_state(f)
        logging.info("Game state saved to %s", state_filename)

    def load_state(self, state_filename="start.save"):
        with open(state_filename, "rb") as f:
            self.pyboy.load_state(f)
        logging.info("Game state loaded from %s", state_filename)

    def close(self):
        self.pyboy.stop()
        logging.info("Emulator stopped.")

    def perform_action(self, action):
        action_map = {
            "A": 'a', "B": 'b', "START": 'start', "SELECT": 'select',
            "UP": 'up', "DOWN": 'down', "LEFT": 'left', "RIGHT": 'right'
        }

        if action in action_map:
            button = action_map[action]
            self._press_button(button, hold_ticks=20, release_ticks=2)
            logging.info("Performed action: %s", action)
        else:
            logging.warning("Unknown action: %s", action)

    def _press_button(self, button, hold_ticks=20, release_ticks=2):
        """Helper method to press and release a button with specified ticks."""
        self.pyboy.button_press(button)
        self.pyboy.tick(hold_ticks)
        self.pyboy.button_release(button)
        self.pyboy.tick(release_ticks)

    def update(self):
        """Retrieve the current state of the game."""
        # Replace with actual game state retrieval logic
        state = {}  # Example placeholder for game state
        logging.info("Game state updated.")
        return state


# # game_controller.py

# import logging
# from pyboy import PyBoy
# from config import ROM_PATH, EMULATION_SPEED  # Ensure these are defined in config

# logging.basicConfig(level=logging.INFO)

# class GameController:
#     def __init__(self, rom_path, emulation_speed=1.0):
#         self.pyboy = PyBoy(rom_path)
#         if not self.pyboy:
#             raise RuntimeError("Failed to initialize PyBoy with the given ROM.")
        
#         self.pyboy.set_emulation_speed(emulation_speed)

#     def save_state(self, state_filename="game_state.save"):
#         with open(state_filename, "wb") as f:
#             self.pyboy.save_state(f)
#         logging.info(f"Game state saved to {state_filename}")

#     def load_state(self, state_filename="start.save"):
#         with open(state_filename, "rb") as f:
#             self.pyboy.load_state(f)
#         logging.info(f"Game state loaded from {state_filename}")

#     def close(self):
#         self.pyboy.stop()
#         logging.info("Emulator stopped.")

#     def perform_action(self, action):
#         action_map = {
#             "A": lambda: self.pyboy.button_press('a'),
#             "B": lambda: self.pyboy.button_press('b'),
#             "START": lambda: self.pyboy.button_press('start'),
#             "SELECT": lambda: self.pyboy.button_press('select'),
#             "UP": lambda: self.pyboy.button_press('up'),
#             "DOWN": lambda: self.pyboy.button_press('down'),
#             "LEFT": lambda: self.pyboy.button_press('left'),
#             "RIGHT": lambda: self.pyboy.button_press('right'),
#         }

#         action_map_release = {
#             "A": lambda: self.pyboy.button_release('a'),
#             "B": lambda: self.pyboy.button_release('b'),
#             "START": lambda: self.pyboy.button_release('start'),
#             "SELECT": lambda: self.pyboy.button_release('select'),
#             "UP": lambda: self.pyboy.button_release('up'),
#             "DOWN": lambda: self.pyboy.button_release('down'),
#             "LEFT": lambda: self.pyboy.button_release('left'),
#             "RIGHT": lambda: self.pyboy.button_release('right'),
#         }

#         button_hold_ticks = 20
#         button_release_ticks = 2

#         if action in action_map:
#             func_action = action_map[action]
#             func_action()
#             self.pyboy.tick(button_hold_ticks)
#             end_func_action = action_map_release[action]
#             end_func_action()
#             self.pyboy.tick(button_release_ticks)
#             logging.info(f"Performed action: {action}")
#         else:
#             logging.warning(f"Unknown action: {action}")

#     def update(self):
#         """Retrieve the current state of the game."""
#         # This example assumes a method to retrieve game state exists.
#         # You will need to replace `get_state_info` with the actual method to retrieve game state data
#         state = {}  # Replace with actual game state retrieval logic
#         return state
