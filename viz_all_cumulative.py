import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_replay_data(replays_dir="replays"):
    """Load all replay files and extract cumulative rewards at each step."""
    all_step_rewards = {}  # step -> list of cumulative rewards at that step

    # Get list of replay files
    replay_files = [f for f in os.listdir(replays_dir) if f.endswith(".pkl")]

    print("Loading replay files...")
    for filename in tqdm(replay_files):
        path = os.path.join(replays_dir, filename)
        with open(path, "rb") as f:
            replay_buffer = pickle.load(f)

        # Track cumulative rewards for each step in this episode
        for step, experience in enumerate(replay_buffer, 1):
            if step not in all_step_rewards:
                all_step_rewards[step] = []
            all_step_rewards[step].append(experience["cumulative_reward"])

    return all_step_rewards


def analyze_distributions(all_step_rewards, max_steps=3000):
    """Calculate statistics for each step."""
    steps = range(1, min(max_steps + 1, max(all_step_rewards.keys()) + 1))

    medians = []
    percentile_25 = []
    percentile_75 = []
    percentile_10 = []
    percentile_90 = []

    print("Analyzing distributions...")
    for step in tqdm(steps):
        if step in all_step_rewards:
            rewards = all_step_rewards[step]
            medians.append(np.median(rewards))
            percentile_25.append(np.percentile(rewards, 25))
            percentile_75.append(np.percentile(rewards, 75))
            percentile_10.append(np.percentile(rewards, 10))
            percentile_90.append(np.percentile(rewards, 90))
        else:
            # Handle missing steps if any
            medians.append(np.nan)
            percentile_25.append(np.nan)
            percentile_75.append(np.nan)
            percentile_10.append(np.nan)
            percentile_90.append(np.nan)

    return steps, medians, percentile_25, percentile_75, percentile_10, percentile_90


def create_visualization(data, output_file="cumulative_rewards_distribution.png"):
    """Create and save the visualization."""
    steps, medians, p25, p75, p10, p90 = data

    plt.figure(figsize=(15, 10))

    # Plot the different percentile ranges
    plt.fill_between(
        steps, p10, p90, alpha=0.2, color="blue", label="10th-90th percentile"
    )
    plt.fill_between(
        steps, p25, p75, alpha=0.3, color="blue", label="25th-75th percentile"
    )
    plt.plot(steps, medians, color="blue", label="Median", linewidth=2)

    plt.xlabel("Step Number")
    plt.ylabel("Cumulative Reward")
    plt.title("Distribution of Cumulative Rewards Across Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()


def main():
    # Load all replay data
    all_step_rewards = load_replay_data()

    # Analyze distributions
    distribution_data = analyze_distributions(all_step_rewards)

    # Create and save visualization
    create_visualization(distribution_data)

    # Print some summary statistics
    final_step = max(all_step_rewards.keys())
    final_rewards = all_step_rewards[final_step]

    print("\nSummary Statistics for Final Step:")
    print(f"Number of episodes: {len(final_rewards)}")
    print(f"Median reward: {np.median(final_rewards):.2f}")
    print(f"Mean reward: {np.mean(final_rewards):.2f}")
    print(f"90th percentile: {np.percentile(final_rewards, 90):.2f}")
    print(f"10th percentile: {np.percentile(final_rewards, 10):.2f}")
    print(f"Max reward: {np.max(final_rewards):.2f}")
    print(f"Min reward: {np.min(final_rewards):.2f}")


if __name__ == "__main__":
    main()
