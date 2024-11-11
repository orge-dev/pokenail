import numpy as np
import pickle
from collections import defaultdict
from actions import Actions
from replay_buffer import ReplayBuffer
import random
import os
import tqdm


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

    def train_from_replays(
        self,
        replays_dir="replays",
        use_cumulative_rewards=False,
        n_experiences=10000000,
        use_combined=True,
    ):
        """Train agent using stored replay experiences"""

        os.makedirs("replays_combined", exist_ok=True)
        combined_path = "replays_combined/latest_combined.pkl"

        all_experiences = []
        if use_combined and os.path.exists(combined_path):
            print(f"Loading combined experiences from {combined_path}")
            with open(combined_path, "rb") as f:
                all_experiences = pickle.load(f)
        else:
            print("Loading individual replay files")
            for filename in tqdm.tqdm(list(os.listdir(replays_dir))):
                if filename.endswith(".pkl"):
                    replay_buffer = ReplayBuffer()
                    replay_buffer.load(os.path.join(replays_dir, filename))
                    assert len(replay_buffer.buffer) > 1
                    exps = [
                        (experience, False) for experience in replay_buffer.buffer[:-1]
                    ] + [(replay_buffer.buffer[-1], True)]
                    all_experiences.extend(exps)

            # Save combined experiences
            print("Saving combined experiences...")
            with open(combined_path, "wb") as f:
                pickle.dump(all_experiences, f)
            print(f"Saved combined experiences to {combined_path}")

        print(f"Collected {len(all_experiences)} experiences")

        print("Training from random sampling of experiences")
        samples = random.sample(
            all_experiences, min(n_experiences, len(all_experiences))
        )
        for experience, is_last_step_of_episode in tqdm.tqdm(samples):
            state = experience["state"]
            action = experience["action"]
            next_state = experience["next_state"]
            step_reward = experience["reward"]
            cumulative_reward = experience["cumulative_reward"]
            # cumulative reward is designed to only count on the last step of the episode
            if use_cumulative_rewards:
                if is_last_step_of_episode:
                    reward = cumulative_reward
                    reward = reward * reward
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
