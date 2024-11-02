import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from actions import Actions


def visualize_q_table(checkpoint_path, out_path):
    # Starting position
    START_POS = (309, 228)

    # Define view window size around start position
    WINDOW_SIZE = 50  # Will show 50x50 area around start

    # Calculate bounds
    y_min = max(0, START_POS[0] - WINDOW_SIZE)
    y_max = START_POS[0] + WINDOW_SIZE
    x_min = max(0, START_POS[1] - WINDOW_SIZE)
    x_max = START_POS[1] + WINDOW_SIZE

    # Load Q-table
    with open(checkpoint_path, "rb") as f:
        q_table_dict = pickle.load(f)
    q_table = defaultdict(lambda: np.zeros(len(Actions.list())), q_table_dict)

    # Create empty map for the window
    action_map = np.zeros((y_max - y_min, x_max - x_min))

    # Action to number mapping
    action_to_num = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "A": 5, "B": 6}

    # Color each position based on highest Q-value action
    for state in q_table:
        print(f"got {state=}")
        # Extract position from state tuple
        for key, value in state:
            if key == "position":
                pos = value
                break

        # Check if position is in our window
        if (y_min <= pos[0] < y_max) and (x_min <= pos[1] < x_max):
            print("including point")
            q_values = q_table[state]
            best_action = Actions.list()[np.argmax(q_values)]
            # Transform coordinates to window space
            window_y = pos[0] - y_min
            window_x = pos[1] - x_min
            action_map[window_y, window_x] = action_to_num[best_action]
        else:
            print("excluding point")

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(action_map)

    # Mark starting position
    start_y = START_POS[0] - y_min
    start_x = START_POS[1] - x_min
    plt.plot(start_x, start_y, "r*", markersize=20, label="Start")  # Using * marker

    # Create legend
    colors = ["black"] + [plt.cm.viridis(x) for x in np.linspace(0, 1, 6)]
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors]
    plt.legend(
        legend_elements
        + [
            plt.Line2D([0], [0], marker="*", color="r", markersize=20, linestyle="None")
        ],
        ["None"] + list(action_to_num.keys()) + ["Start Position"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )  # Move legend outside plot

    plt.title("Most Rewarded Actions per Position (Around Start)")
    plt.savefig(out_path)
    plt.close()


def main():
    import os

    # Create output directory
    out_dir = "q_table_visualizations"
    os.makedirs(out_dir, exist_ok=True)

    # Process all checkpoint files
    checkpoints_dir = "checkpoints"
    for filename in os.listdir(checkpoints_dir):
        if filename.startswith("agent_state_") and filename.endswith(".pkl"):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            out_path = os.path.join(
                out_dir, f"qtable_viz_{filename.replace('.pkl', '.png')}"
            )
            print(f"Processing {filename}...")
            visualize_q_table(checkpoint_path, out_path)
            print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
