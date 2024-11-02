import numpy as np
import pickle
from collections import defaultdict
from actions import Actions


class AIAgent:
    def __init__(
        self,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        num_actions = len(Actions.list())
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
        state = tuple(state.items())
        if np.random.random() < self.exploration_rate:
            return np.random.choice(Actions.list())
        action_index = np.argmax(self.q_table[state])
        return Actions.list()[action_index]

    def save_state(self, filename="agent_state.pkl"):
        """Saves the Q-table."""
        with open(filename, "wb") as file:
            pickle.dump(dict(self.q_table), file)
            print(f"saved state to {filename=}")

    def load_state(self, filename="agent_state.pkl"):
        """Loads the Q-table."""
        with open(filename, "rb") as file:
            self.q_table = defaultdict(
                lambda: np.zeros(len(Actions.list())), pickle.load(file)
            )
