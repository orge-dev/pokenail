import numpy as np
import pickle
from collections import defaultdict
from actions import Actions  # Ensure Actions enum is imported
from env import env_red  # Import your environment
import logging


class AIAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        num_actions = len(Actions.list())
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
        state = tuple(state.items())
        action_index = np.argmax(self.q_table[state])
        return Actions.list()[action_index]

    def update(self, state, action, reward, next_state):
        """Updates the Q-table using the Q-learning update rule."""
        state, next_state = tuple(state.items()), tuple(next_state.items())
        action_index = Actions.list().index(action)
        best_next_action_value = np.max(self.q_table[next_state])

        self.q_table[state][action_index] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_action_value
            - self.q_table[state][action_index]
        )

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def run(self, env, checkpoint="q_table_final.pkl", max_steps=100):
        """Run the agent in the environment using the Q-table from the checkpoint file."""
        self.load_q_table(checkpoint)
        state = env.reset()
        done, step = False, 0

        while not done and step < max_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            logging.info(f"Step {step}: Action: {action}, Reward: {reward}")
            state = next_state
            step += 1
        logging.info("Run completed.")

    def save_q_table(self, filename="q_table.pkl"):
        """Saves the Q-table to a file."""
        with open(filename, "wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load_q_table(self, filename="q_table.pkl"):
        """Loads the Q-table from a file."""
        with open(filename, "rb") as file:
            self.q_table = defaultdict(
                lambda: np.zeros(len(Actions.list())), pickle.load(file)
            )


# Training Loop
if __name__ == "__main__":
    env = env_red()  # Initialize the environment
    agent = AIAgent()
    total_episodes = 1000
    max_steps_per_episode = 100

    for episode in range(total_episodes):
        state = env.reset()
        done, step = False, 0

        while not done and step < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            step += 1

        if episode % 100 == 0:
            agent.save_q_table(f"q_table_checkpoint_{episode}.pkl")
            logging.info(f"Checkpoint saved at episode {episode}")

    agent.save_q_table("q_table_final.pkl")
    logging.info("Training completed. Final Q-table saved.")
