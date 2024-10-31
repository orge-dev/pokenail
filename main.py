import random
from actions import Actions
from ai_agent import AIAgent
from config import ROM_PATH, EMULATION_SPEED, MAX_STEPS
from game_controller import GameController
from env import env_red
import game_controller
import argparse

def main():
    # Initialize environment and AI agent
    environment = env_red()
    ai_agent = AIAgent()
    state = environment.reset()
    parser = argparse.ArgumentParser(description="Run Pokémon Red with AI or manual control.")
    parser.add_argument("--manual", action="store_true", help="Enable manual control mode.")
    args = parser.parse_args()

    try:
        if not args.manual:
            while True:
                # Select and perform action
                action = ai_agent.select_action(state)
                next_state, reward, done, _ = environment.step(action)

                # Update AI agent and state
                ai_agent.update(state, action, reward, next_state)
                state = next_state
                # Exit loop if game is over
                if done:
                    break
        else:
            try:
             #pyboy = PyBoy(ROM_PATH)
             controller = GameController(ROM_PATH, EMULATION_SPEED)  
             controller.load_state()
             while True:
 
              if not controller.pyboy.tick():
               break  # Exit if the emulator signals to stop

            except KeyboardInterrupt:
                print("Program interrupted. Stopping emulator...")

            finally:
                # Clean up and close the emulator
                pyboy.stop()
                    #else we want the user to input


    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()

if __name__ == "__main__":
    main()
