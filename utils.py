import datetime
import random
import string
import time
import os
import pickle

from viz import visualize_heatmap, visualize_path


def generate_timestamped_id():
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

    # Generate random string of letters and digits
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    return timestamp + random_str


def display_episode_statistics(episodes_dir="episodes"):
    """Read and display all episode statistics in the directory."""
    print("\nEpisode Statistics:")
    print("-" * 50)
    for filename in sorted(os.listdir(episodes_dir)):
        if filename.startswith("episode_") and filename.endswith(".pkl"):
            filepath = os.path.join(episodes_dir, filename)
            try:
                with open(filepath, "rb") as f:
                    stats = pickle.load(f)

                    battle_info = "No battle found"
                    if stats["steps_to_battle"] is not None:
                        battle_info = f"Battle found at step {stats['steps_to_battle']}"

                    print(f"\nEpisode {filename}:")
                    print(f"Unique coordinates visited: {len(stats['visited_coords'])}")
                    print(f"Total steps: {stats['total_steps']}")
                    print(f"Total reward: {stats['total_reward']}")
                    print(f"Battle status: {battle_info}")
            except (EOFError, pickle.UnpicklingError):
                print(f"Could not read {filepath}. It may be corrupted or incomplete.")


def monitor_episodes(episodes_dir="episodes", polling_interval=5):
    """Continuously monitor the episodes directory for new or modified files and print statistics."""
    processed_files = {}

    print("\nEpisode Statistics Monitor:")
    print("-" * 50)

    while True:
        updated = False  # Flag to check if new data has been found
        for filename in sorted(os.listdir(episodes_dir)):
            if filename.startswith("episode_") and filename.endswith(".pkl"):
                filepath = os.path.join(episodes_dir, filename)

                # Get the last modified time of the file
                last_modified_time = os.path.getmtime(filepath)

                # If it's a new or modified file, update statistics
                if (
                    filepath not in processed_files
                    or processed_files[filepath] < last_modified_time
                ):
                    processed_files[filepath] = last_modified_time
                    updated = True

        # Display statistics only if there's an update
        if updated:
            os.system(
                "clear"
            )  # Clear the console output for a refreshed display (use 'cls' on Windows)
            display_episode_statistics(episodes_dir)

        # Wait before checking again
        time.sleep(polling_interval)


# In utils.py, modify the analyze_episodes function:

def analyze_episodes(episodes_dir="episodes", skip_viz=False):
    """Read all episode files and print statistics."""
    print("\nEpisode Statistics:")
    print("-" * 50)

    battles_found = 0
    total_episodes = 0
    best_battle_steps = float("inf")
    best_battle_reward = 0
    best_episode = None

    for filename in sorted(os.listdir(episodes_dir)):
        if filename.startswith("episode_") and filename.endswith(".pkl"):
            total_episodes += 1
            filepath = os.path.join(episodes_dir, filename)
            try:
                with open(filepath, "rb") as f:
                    stats = pickle.load(f)

                    if not skip_viz:
                        # Create visualizations directory if it doesn't exist
                        os.makedirs("visualizations", exist_ok=True)

                        # Generate visualization for this episode
                        path_save = f"visualizations/{filename.replace('.pkl', '_path.png')}"
                        heat_save = f"visualizations/{filename.replace('.pkl', '_heatmap.png')}"
                        visualize_path(stats, path_save)
                        visualize_heatmap(stats, heat_save)
                        print(f"Path visualization saved to {path_save}")
                        print(f"Heatmap visualization saved to {heat_save}")

                    if stats["steps_to_battle"] is not None:
                        battles_found += 1
                        if stats["steps_to_battle"] < best_battle_steps:
                            best_battle_steps = stats["steps_to_battle"]
                            best_battle_reward = stats["total_reward"]
                            best_episode = filename

                    if not skip_viz:
                        print(f"\nEpisode {filename}:")
                        print(f"Total steps: {stats['total_steps']}")
                        print(f"Final position: {stats['final_position']}")
                        print(f"Total unique positions visited: {len(stats['visited_coords'])}")
                        print(f"Battle found: {stats['battle']}")
                        print(f"Battle reward applied: {stats['battle_reward_applied']}")
                        print(f"Last distance reward: {stats['last_distance_reward']}")
                        print(f"Total reward: {stats['total_reward']:.2f}")

                        if stats["steps_to_battle"] is not None:
                            print(f"Battle found at step {stats['steps_to_battle']}")
                        else:
                            print("No battle found")
            except EOFError:
                pass

    print("\nSummary:")
    print(f"Total episodes: {total_episodes}")
    print(f"Battles found: {battles_found}")
    print(f"Success rate: {(battles_found/total_episodes)*100:.1f}%")
    if best_episode:
        print("\nBest performance:")
        print(f"Episode: {best_episode}")
        print(f"Steps to battle: {best_battle_steps}")
        print(f"Reward: {best_battle_reward:.2f}")

# And modify the main block:

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze episode statistics")
    parser.add_argument("--viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()
    
    analyze_episodes(skip_viz=not args.viz)
