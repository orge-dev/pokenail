import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_path(stats, save_path=None):
    """
    Visualize the path taken during an episode.
    Args:
        stats: Episode statistics dictionary
        save_path: Optional path to save the plot
    """
    visited_coords = stats["visited_coords"]
    final_pos = stats["final_position"]
    target_pos = (309, 99)  # Hardcoded target position

    # Extract x and y coordinates
    y_coords, x_coords = zip(*visited_coords)

    # Create figure and axis
    plt.figure(figsize=(12, 12))

    # Create color gradient based on order of visits
    colors = np.linspace(0, 1, len(x_coords))
    cmap = LinearSegmentedColormap.from_list("custom", ["blue", "red"])

    # Plot path with color gradient
    plt.scatter(x_coords, y_coords, c=colors, cmap=cmap, alpha=0.5, s=10)

    # Plot start point (first coordinate)
    plt.plot(x_coords[0], y_coords[0], "go", markersize=15, label="Start")

    # Plot end point (final position)
    plt.plot(final_pos[1], final_pos[0], "ro", markersize=15, label="End")

    # Plot target position
    plt.plot(target_pos[1], target_pos[0], "yo", markersize=15, label="Target")

    # Add labels and title
    plt.title(f'Path Visualization\nTotal Steps: {stats["total_steps"]}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# TODO: Maybe combine heatmaps from different episodes into one heatmap
def visualize_heatmap(stats, save_path=None):
    """
    Create a heatmap visualization of visited coordinates.
    Args:
        stats: Episode statistics dictionary
        save_path: Optional path to save the plot
    """
    visited_coords = stats["visited_coords"]

    # Create 2D histogram
    y_coords, x_coords = zip(*visited_coords)

    # Create 2D histogram with appropriate bins
    heatmap, xedges, yedges = np.histogram2d(
        y_coords,
        x_coords,
        bins=50,
        range=[[0, 484], [0, 476]],  # Based on GLOBAL_MAP_SHAPE
    )

    # Create figure
    plt.figure(figsize=(12, 12))

    # Plot heatmap
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    # Add colorbar
    plt.colorbar(label="Visit Count")

    # Plot start, end and target positions
    plt.plot(
        visited_coords[0][1], visited_coords[0][0], "go", markersize=15, label="Start"
    )
    plt.plot(
        stats["final_position"][1],
        stats["final_position"][0],
        "ro",
        markersize=15,
        label="End",
    )
    plt.plot(99, 309, "yo", markersize=15, label="Target")  # Hardcoded target

    # Add labels and title
    plt.title(f'Visit Density Heatmap\nTotal Steps: {stats["total_steps"]}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
