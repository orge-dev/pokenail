import random
from actions import Actions
from ai_agent import AIAgent
from config import ROM_PATH, EMULATION_SPEED, MAX_STEPS
from env import env_red

def main():
    # Initialize environment and AI agent
    environment = env_red()
    ai_agent = AIAgent()
    state = environment.reset()

    try:
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

    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()

if __name__ == "__main__":
    main()
