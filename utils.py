import os
import pickle
import time
import datetime
import random
import string
def generate_timestamped_id():
    """Generate a unique ID with a timestamp and a random string."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
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
                if filepath not in processed_files or processed_files[filepath] < last_modified_time:
                    processed_files[filepath] = last_modified_time
                    updated = True

        # Display statistics only if there's an update
        if updated:
            os.system('clear')  # Clear the console output for a refreshed display (use 'cls' on Windows)
            display_episode_statistics(episodes_dir)

        # Wait before checking again
        time.sleep(polling_interval)


if __name__ == "__main__":
    monitor_episodes()
