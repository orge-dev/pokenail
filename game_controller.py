from pyboy import PyBoy
from config import ROM_PATH  # Import ROM_PATH from config
# from action import Action

class GameController:
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path)
        if not self.pyboy:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM.")
    #add a save state 

    def start(self):
       
        try:
            while True:
                # Update the emulator state
                if not self.pyboy.tick():
                    break  # Exit if the emulator signals to stop
        except Exception as e:
            print(f"Error during update: {e}")
            return None

    # def perform_action(self, action: Action):
    #     try:
    #         action.perform(self.pyboy)
    #     except Exception as e:
    #         print(f"Error performing action: {e}")

    def close(self):
        self.pyboy.stop()

# Main game loop
if __name__ == "__main__":
    controller = GameController(ROM_PATH)
    try:
        while True:
            game_state = controller.start()
            if game_state is None:
                break
            # Process game_state as needed
    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        controller.close()
