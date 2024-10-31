import argparse
import random
from actions import Actions
from ai_agent import AIAgent
from config import ROM_PATH, EMULATION_SPEED, MAX_STEPS
from env import env_red

def get_user_action():
    """Prompt user for action input."""
    print("Enter action (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT):")
    action = input().upper()
    valid_actions = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]
    if action in valid_actions:
        return action
    else:
        print("Invalid action. Try again.")
        return get_user_action()

def main():
    parser = argparse.ArgumentParser(description="Run Pokémon Red with AI or manual control.")
    parser.add_argument("--manual", action="store_true", help="Enable manual control mode.")
    args = parser.parse_args()

    # Initialize environment and AI agent
    environment = env_red()
    ai_agent = AIAgent()
    state = environment.reset()

    try:
        if not args.manual:
            print("Manual control mode enabled. Use keyboard to control the game.")
            while True:
                # Get user input for action
                action = get_user_action()
                next_state, reward, done, _ = environment.step(action)
                
                # Exit loop if game is over
                if done:
                    break
        else:
            print("AI control mode enabled.")
            while True:
                # Select and perform action using AI
                action = ai_agent.select_action(state)
                next_state, reward, done, _ = environment.step(action)
                
                # Update AI agent and state
                ai_agent.update(state, action, reward, next_state)
                state = next_state

                # Exit loop if game is over
                if done:
                    break

    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()

if __name__ == "__main__":
    main()
