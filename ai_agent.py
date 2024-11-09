import numpy as np
import pickle
from collections import defaultdict
from actions import Actions
from replay_buffer import ReplayBuffer
import os


class AIAgent:
    def __init__(
        self,
        learning_rate=0.05,
        discount_factor=0.9,
        # 1.0 = random, not using q table
        # Setting this .8 (not very low) seems to make agent much prefer
        # walking upwards in starting building than 1.0 why... maybe not enough episodes/due to sparse reward (lookup reward shaping? or go straight to ppo?)
        exploration_rate=1.0,
    ):
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
        if np.random.random() < self.exploration_rate:
            return np.random.choice(Actions.list())
        action_index = np.argmax(self.q_table[state])
        return Actions.list()[action_index]

    def update_q_table(self, state, action, next_state, reward):
        """Updates Q-table using Q-learning algorithm"""
        # Convert dict states to tuples for hashing
        # Get index of the action taken
        action_index = Actions.list().index(action)

        # Get max Q-value for next state
        best_next_action_value = np.max(self.q_table[next_state])

        # Q-learning update formula
        current_q = self.q_table[state][action_index]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_action_value - current_q
        )
        self.q_table[state][action_index] = new_q

    def train_from_replays(self, replays_dir="replays", use_cumulative_rewards=True):
        """Train agent using stored replay experiences"""
        print("Training from replay files...")
        for filename in os.listdir(replays_dir):
            if filename.endswith(".pkl"):
                replay_path = os.path.join(replays_dir, filename)
                replay_buffer = ReplayBuffer()
                replay_buffer.load(replay_path)

                print(f"Training from {filename}...")

                experiences_reversed = replay_buffer.buffer[::-1]
                for exp_i, experience in enumerate(experiences_reversed):
                    state = experience["state"]
                    action = experience["action"]
                    next_state = experience["next_state"]
                    step_reward = experience["reward"]
                    cumulative_reward = experience["cumulative_reward"]
                    # cumulative reward is designed to only count on the last step of the episode
                    if use_cumulative_rewards:
                        if exp_i == 0:  
                            reward = cumulative_reward
                        else:
                            reward = 0
                    else:
                        reward = step_reward
                    self.update_q_table(state, action, next_state, reward)

    def save_state(self, filename="agent_state.pkl", do_print=False):
        """Saves the Q-table."""
        with open(filename, "wb") as file:
            pickle.dump(dict(self.q_table), file)
            if do_print:
                print(f"Saved AI state to {filename}")

    def load_state(self, filename="agent_state.pkl"):
        """Loads the Q-table."""
        with open(filename, "rb") as file:
            self.q_table = defaultdict(
                lambda: np.zeros(len(Actions.list())), pickle.load(file)
            )
