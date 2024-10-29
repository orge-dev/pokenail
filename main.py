import random
from game_controller import GameController
from actions import Actions  # Import the Actions enum
# from ai_agent import AIAgent  # Uncomment this when you have the AIAgent defined
from config import ROM_PATH, EMULATION_SPEED  # Ensure these are correctly defined in config

def main():
    controller = GameController(ROM_PATH, EMULATION_SPEED)  # Initialize the controller
    # ai_agent = AIAgent()  # Uncomment and define when ready

    # Load the saved state before starting the game
    controller.load_state()  

    still_running = True

    try:    
        while still_running:
            # Update game state
            state = controller.update()  # Implement this method in GameController
            
            # Get a random action
            action = random.choice(Actions.list())  # Randomly select an action
            
            # Perform action based on AI's decision
            controller.perform_action(action)  # Implement this method
            #controller.perform_action_a(action)
            # Start the game loop
            still_running = controller.step()

    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
         controller.save_state()  # Save the state before closing
         controller.close()  # Close the emulator

if __name__ == "__main__":
    main()
