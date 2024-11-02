from ai_agent import AIAgent
from env import env_red
from utils import generate_timestamped_id
import argparse
import os
from multiprocessing import Pool, cpu_count


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Pokémon Red with AI or manual control."
    )
    parser.add_argument(
        "--manual", action="store_true", help="Enable manual control mode."
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--episode_length", type=int, default=3000, help="Steps per episode"
    )
    parser.add_argument(
        "--train_from_replays",
        action="store_true",
        help="Train agent using stored replay experiences",
    )
    parser.add_argument(
        "--processes", type=int, default=1, 
        help="Number of parallel processes (default: CPU count)"
    )
    return parser.parse_args()


def run_ai_mode(
    environment, episode_id=None, previous_episode_id=None, episode_length=1000
):
    ai_agent = AIAgent()

    # Load previous episode's checkpoint if exists
    if previous_episode_id:
        checkpoint = f"checkpoints/agent_state_{previous_episode_id}.pkl"
        if os.path.exists(checkpoint):
            ai_agent.load_state(checkpoint)

    # Generate new episode ID if none provided
    if episode_id is None:
        episode_id = generate_timestamped_id()

    state = environment.reset()
    step = 0
    while step < episode_length:
        step += 1
        if step % 100 == 0:  # Save checkpoint less frequently
            ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")

        action = ai_agent.select_action(state)
        next_state, reward, done, _ = environment.step(action, False)

        # debug logs
        if (
            step % 1000 == 0 or step == episode_length
        ):  # Every 1000 steps or at episode end
            current_pos = tuple(next_state["position"])
            distance, distance_reward = environment.calculate_distance_metrics(
                current_pos
            )

            print(f"\nEpisode step {step}/{episode_length}:")
            print(f"Total reward: {environment.total_reward:.2f}")
            print(f"Distance: {distance:.2f}")
            print(f"Distance reward: {distance_reward:.2f}")
            print(f"Current position: {current_pos}")

        state = next_state
        if done:
            break

    # Final save at episode end
    ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")
    environment.save_episode_stats(episode_id)
    return episode_id


def run_manual_mode():
    environment = env_red()
    environment.reset()
    done = False
    while not done:
        next_state, reward, done, _ = environment.step(manual=True)


def run_episode(args):
    episode_num, episode_length, headless, initial_q_state = args
    environment = env_red(headless=headless)

    try:
        print(f"\n*** Starting episode {episode_num}")
        # add episode num to filename in case timestamps collide when running in parallel
        episode_id = f"{generate_timestamped_id()}_ep{episode_num}"
        ai_agent = AIAgent()

        if initial_q_state and os.path.exists(initial_q_state):
            ai_agent.load_state(initial_q_state)

        state = environment.reset()
        step = 0
        while step < episode_length:
            step += 1
            if step % 100 == 0:
                ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")

            action = ai_agent.select_action(state)
            next_state, reward, done, _ = environment.step(action, False)

            if step % 1000 == 0 or step == episode_length:
                current_pos = tuple(next_state["position"])
                distance, distance_reward = environment.calculate_distance_metrics(
                    current_pos
                )
                print(f"\nEpisode {episode_num} step {step}/{episode_length}:")
                print(f"Total reward: {environment.total_reward:.2f}")
                print(f"Current position: {current_pos}")

            state = next_state
            if done:
                break

        ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")
        environment.save_episode_stats(episode_id)
        return episode_id

    finally:
        environment.close()


def main():
    args = parse_arguments()
    environment = env_red(headless=args.headless)

    try:
        initial_q_state = "checkpoints/agent_state_20241103_190121_Qn3O6CK9.pkl"
        # change to None to start with blank q table (doesnt apply to manual mode)
        # initial_q_state = None

        if args.train_from_replays:
            agent = AIAgent()
            if initial_q_state:
                agent.load_state(initial_q_state)
            agent.train_from_replays()
            agent.save_state(
                f"checkpoints/agent_state_{generate_timestamped_id()}.pkl",
                do_print=True,
            )

        elif args.manual:
            run_manual_mode()

        else:
            # Parallel processing for AI episodes
            num_processes = args.processes or cpu_count()
            print(f"Running {args.episodes} episodes using {num_processes} processes")

            episode_args = [
                (i + 1, args.episode_length, args.headless, initial_q_state)
                for i in range(args.episodes)
            ]

            with Pool(processes=num_processes) as pool:
                episode_ids = pool.map(run_episode, episode_args, chunksize=1)

            print("\nCompleted episodes:", len(episode_ids))
    except KeyboardInterrupt:
        print("Program interrupted. Stopping emulator...")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
