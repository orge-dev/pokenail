import random
import string
import datetime
import os
import pickle


def generate_timestamped_id():
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

    # Generate random string of letters and digits
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    return timestamp + random_str


def analyze_episodes(episodes_dir="episodes"):
    """Read all episode files and print statistics."""
    print("\nEpisode Statistics:")
    print("-" * 50)

    for filename in sorted(os.listdir(episodes_dir)):
        if filename.startswith("episode_") and filename.endswith(".pkl"):
            filepath = os.path.join(episodes_dir, filename)
            print(filepath)
            try:
                with open(filepath, "rb") as f:
                    stats = pickle.load(f)

                    battle_info = "No battle found"
                    if stats["steps_to_battle"] is not None:
                        battle_info = f"Battle found at step {stats['steps_to_battle']}"

                    print(f"\nEpisode {filename}:")
                    print(f"Unique coordinates visited: {len(stats['visited_coords'])}")
                    print(f"Total steps: {stats['total_steps']}")
                    print(f"Battle status: {battle_info}")
            except EOFError:
                pass


if __name__ == "__main__":
    analyze_episodes()
