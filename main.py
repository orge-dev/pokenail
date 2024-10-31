import random
from actions import Actions  # Import the Actions enum
from ai_agent import AIAgent
from config import ROM_PATH, EMULATION_SPEED, MAX_STEPS
from env import env_red  # Use env_red as the environment

def main():
    environment = env_red()  # Initialize the environment (which includes GameController)
    ai_agent = AIAgent()  # Initialize your AI agent

    # Reset the environment to get the initial state
    state = environment.reset()

    try:    
        while True:
            # Get an action from the AI agent based on the current state
            action = ai_agent.select_action(state)

            # Perform the action in the environment
            next_state, reward, done, _ = environment.step(action)

            # Update the AI agent with the new experience (optional, for training)
            ai_agent.update(state, action, reward, next_state)

            # Update state
            state = next_state

            # Check if the game is over
            if done:
                break

    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()  # Close the environment safely

if __name__ == "__main__":
    main()
