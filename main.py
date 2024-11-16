import argparse
import concurrent.futures
import os
from multiprocessing import Manager, Pool, cpu_count

from ai_agent import AIAgent, evaluate_training_progress
from env import EnvRed
from utils import generate_timestamped_id
from web_viz import WebViz


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Pokémon Red with AI or manual control."
    )
    parser.add_argument(
        "--manual", action="store_true", help="Enable manual control mode."
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to run"
    )
    parser.add_argument(
        "--episode_length", type=int, default=5000, help="Steps per episode"
    )
    parser.add_argument(
        "--train_from_replays",
        action="store_true",
        help="Train agent using stored replay experiences",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: CPU count)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Path to initial agent state (Q table) file to load",
        default=None,
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Path to directory with agent states to evaluate",
        default=None,
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
        next_state, _, done, _ = environment.step(action, False)

        # debug logs
        if (
            step % 1000 == 0 or step == episode_length
        ):  # Every 1000 steps or at episode end
            current_pos = next_state.position
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
    environment = EnvRed()
    environment.reset()
    done = False
    step = 0
    while not done:
        step += 1
        environment.step(manual=True)
        if step % 50 == 0:
            environment.controller.save_state()


def run_episode(args, environment=None, exploration_rate=1.0):
    episode_num, episode_length, headless, agent_file, multiprocess_dict = args
    if environment is None:
        should_close_environment = True
        environment = EnvRed(headless=headless)
    else:
        should_close_environment = False

    web_viz = WebViz()
    background_tasks = set()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as thread_pool:
            print(f"\n*** Starting episode {episode_num}")
            # add episode num to filename in case timestamps collide when running in parallel
            episode_id = f"{generate_timestamped_id()}_ep{episode_num}"
            ai_agent = AIAgent(
                exploration_rate=exploration_rate, multiprocess_dict=multiprocess_dict
            )

            if agent_file and os.path.exists(agent_file):
                ai_agent.load_state(agent_file)

            environment.reset()
            state = environment.state_for_agent()
            step = 0
            final_cumulative_reward = None
            while step < episode_length:
                step += 1
                if step % 100 == 0:
                    ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")

                action = ai_agent.select_action(state)
                next_state, _, cumulative_reward, done, _ = environment.step(
                    action, False
                )
                final_cumulative_reward = cumulative_reward  # Update the final value

                if step % 1000 == 0 or step == episode_length:
                    current_pos = next_state.position
                    print(f"\nEpisode {episode_num} step {step}/{episode_length}:")
                    print(f"Total reward: {environment.total_reward:.2f}")
                    print(f"Current position: {current_pos}")

                fut = thread_pool.submit(
                    web_viz.broadcast_position, environment.local_position
                )
                background_tasks.add(fut)
                fut.add_done_callback(background_tasks.discard)

                state = next_state
                if done:
                    break

            print(
                f"\nEpisode {episode_num} finished with cumulative reward: {final_cumulative_reward}"
            )
            ai_agent.save_state(f"checkpoints/agent_state_{episode_id}.pkl")
            environment.save_episode_stats(episode_id)

            return episode_id

    finally:
        if should_close_environment:
            environment.close()


def main():
    args = parse_arguments()

    agent_file = args.agent

    # Setup folders where we save stuff
    os.makedirs("checkpoints/from_replays", exist_ok=True)

    if args.train_from_replays:
        agent = AIAgent()
        if agent_file:
            agent.load_state(agent_file)
        agent_id = generate_timestamped_id()
        agent.train_from_replays(agent_id)
        q_state_filename = f"checkpoints/from_replays/agent_state_{agent_id}.pkl"
        agent.save_state(q_state_filename, do_print=True)

        print("\nEvaluating final agent...")
        run_episode(
            (1, 2000, False, q_state_filename, None),
            exploration_rate=0.2,
        )

        print("\nEvaluating training progress...")
        evaluate_training_progress()

    elif args.eval:
        evaluate_training_progress(args.eval)

    elif args.manual:
        run_manual_mode()

    else:
        # Run directly if single process and not headless
        if args.processes == 1 and not args.headless:
            environment = EnvRed()
            print(f"Running {args.episodes} episodes sequentially")
            episode_ids = []
            try:
                for i in range(args.episodes):
                    episode_id = run_episode(
                        (i + 1, args.episode_length, args.headless, agent_file, None),
                        environment=environment,
                        # exploration_rate=0.2,  # use q table when not headless, so we see AI actions
                    )
                    episode_ids.append(episode_id)
            finally:
                environment.close()  # Only close environment after all episodes
        else:
            with Manager() as manager:
                multiprocess_dict = manager.dict()
                # Parallel processing for multiple processes or headless mode
                num_processes = args.processes or cpu_count()
                print(
                    f"Running {args.episodes} episodes using {num_processes} processes"
                )

                episode_args = [
                    (
                        i + 1,
                        args.episode_length,
                        args.headless,
                        agent_file,
                        multiprocess_dict,
                    )
                    for i in range(args.episodes)
                ]

                with Pool(processes=num_processes) as pool:
                    episode_ids = pool.map(run_episode, episode_args, chunksize=1)

        print("\nCompleted episodes:", len(episode_ids))


if __name__ == "__main__":
    main()
