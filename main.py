from ai_agent import AIAgent
from env import env_red
from utils import generate_timestamped_id
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Pokémon Red with AI or manual control."
    )
    parser.add_argument(
        "--manual", action="store_true", help="Enable manual control mode."
    )
    return parser.parse_args()


def run_ai_mode(checkpoint=None):
    environment = env_red()
    # Uncomment this to start the run from a checkpoint instead of an empty q table
    # checkpoint="checkpoints/agent_state_20241101_173109_ahuDmaYL.pkl"
    environment.reset()
    ai_agent = AIAgent()
    if checkpoint is not None:
        ai_agent.load_state(checkpoint)

    episode_id = generate_timestamped_id()

    state = environment.reset()
    step = 0
    while True:
        step += 1
        if step % 1000 == 0:
            ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")
        action = ai_agent.select_action(state)
        # environment.controller.perform_action(action)
        next_state, reward, done, _ = environment.step(action, False)
        print(f"{next_state=}, {reward=}, {done=}")
        state = next_state
        if done:
            break


def run_manual_mode():
    environment = env_red()
    environment.reset()  # Ensure previous_state is initialized
    controller = environment.controller
    try:
        controller.load_state()
        done = False
        while not done:
            # Call step without an action, only in manual mode
            next_state, reward, done, _ = environment.step(manual=True)
    finally:
        controller.save_state()
        controller.close()


def main():
    args = parse_arguments()
    environment = env_red()
    try:
        if args.manual:
            run_manual_mode()
        else:
            run_ai_mode()
    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
