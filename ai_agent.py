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
        self.visited_coords = set()  # Track visited coordinates

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
        state = tuple(state.items())
        r = np.random.random()
        if r < self.exploration_rate:
            return np.random.choice(Actions.list())
        action_index = np.argmax(self.q_table[state])
        #print(f"{r=} {self.exploration_rate=} q table", self.q_table[state])
        #print(state)
        return Actions.list()[action_index]

    def update(self, state, action, reward, next_state):
        """Updates the Q-table using the Q-learning update rule."""
        position = state['position']
        position_tuple = tuple(position)
        
        # Add exploration reward if position is new
        exploration_reward = 10 if position_tuple not in self.visited_coords else 0
        total_reward = reward + exploration_reward
        print(f"{exploration_reward=}")
        
        # Add position to visited set
        self.visited_coords.add(position_tuple)

        state, next_state = tuple(state.items()), tuple(next_state.items())
        action_index = Actions.list().index(action)
        best_next_action_value = np.max(self.q_table[next_state])

        self.q_table[state][action_index] += self.learning_rate * (
            total_reward
            + self.discount_factor * best_next_action_value
            - self.q_table[state][action_index]
        )

        # Encourages exploration only/mostly at beginning of episode
        # if self.exploration_rate > self.min_exploration_rate:
        #     self.exploration_rate *= self.exploration_decay

    def run(self, env, checkpoint="agent_state.pkl", max_steps=100):
        """Run the agent in the environment using the Q-table from the checkpoint file."""
        self.load_state(checkpoint)
        state = env.reset()
        done, step = False, 0

        while not done and step < max_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            logging.info(f"Step {step}: Action: {action}, Reward: {reward}")
            state = next_state
            step += 1
        logging.info("Run completed.")

    def save_state(self, filename="agent_state.pkl"):
        """Saves both Q-table and visited coordinates"""
        state = {
            'q_table': dict(self.q_table),
            'visited_coords': self.visited_coords
        }
        with open(filename, "wb") as file:
            pickle.dump(state, file)
            print(f"saved state to {filename=}")

    def load_state(self, filename="agent_state.pkl"):
        """Loads both Q-table and visited coordinates"""
        with open(filename, "rb") as file:
            state = pickle.load(file)
            self.q_table = defaultdict(
                lambda: np.zeros(len(Actions.list())), 
                state['q_table']
            )
            self.visited_coords = state['visited_coords']


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
