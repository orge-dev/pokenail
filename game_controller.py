from pyboy import PyBoy
from config import ROM_PATH, EMULATION_SPEED  # Import ROM_PATH and EMULATION_SPEED from config

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
        except Exception as e:
            print(f"Error during update: {e}")

    def save_state(self, state_filename="game_state.save"):
        with open(state_filename, "wb") as f:
            self.pyboy.save_state(f)
        print(f"Game state saved to {state_filename}")

    def load_state(self, state_filename="game_state.save"):
        with open(state_filename, "rb") as f:
            self.pyboy.load_state(f)
        print(f"Game state loaded from {state_filename}")

    def close(self):
        self.pyboy.stop()

if __name__ == "__main__":
    controller = GameController(ROM_PATH, EMULATION_SPEED)  # Initialize the controller

    # Load the saved state before starting the game
    controller.load_state()  

    try:    
        controller.start()  # Start the game loop
    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        controller.save_state()  # Save the state before closing
        controller.close()  # Close the emulator
