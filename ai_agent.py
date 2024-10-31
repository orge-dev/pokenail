import random
import numpy as np
import pickle
from collections import defaultdict
from actions import Actions  # Ensure Actions enum is imported
from env import env_red  # Import your environment
from pathlib import Path
import uuid
from stable_baselines3 import PPO

class AIAgent:
    # def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
    #     # Initialize parameters
    #     num_actions = len(Actions.list())  # Get the number of actions
    #     self.q_table = defaultdict(lambda: np.zeros(num_actions))  # Q-table for state-action values
    #     self.learning_rate = learning_rate  # Alpha
    #     self.discount_factor = discount_factor  # Gamma
    #     self.exploration_rate = exploration_rate  # Epsilon
    #     self.exploration_decay = exploration_decay  # Decay for epsilon
    #     self.min_exploration_rate = min_exploration_rate  # Minimum epsilon value

    # def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
    #     # Initialize parameters
    #     num_actions = len(Actions.list())  # Get the number of actions
    #     self.q_table = defaultdict(lambda: np.zeros(num_actions))  # Q-table for state-action values
    #     self.learning_rate = learning_rate  # Alpha
    #     self.discount_factor = discount_factor  # Gamma
    #     self.exploration_rate = exploration_rate  # Epsilon
    #     self.exploration_decay = exploration_decay  # Decay for epsilon
    #     self.min_exploration_rate = min_exploration_rate  # Minimum epsilon value
  
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        num_actions = len(Actions.list())  # Get the number of actions
        self.q_table = defaultdict(lambda: np.zeros(num_actions))  # Q-table for state-action values
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay  # Decay for epsilon
        self.min_exploration_rate = min_exploration_rate  # Minimum epsilon value

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
        state = tuple(state.items())  # Convert state dict to a hashable tuple
        
        # Get the index of the action with the highest Q-value
        action_index = np.argmax(self.q_table[state])
        
        # Return the action based on the index
        return Actions.list()[action_index]

    def update(self, state, action, reward, next_state):
        """Updates the Q-table using the Q-learning update rule."""
        state = tuple(state.items())  # Convert state dict to a hashable tuple
        next_state = tuple(next_state.items())  # Convert next_state dict to a hashable tuple

        action_index = Actions.list().index(action)  # Get the index of the action
        best_next_action_value = np.max(self.q_table[next_state])  # Best Q-value for next state

        # Q-learning formula update
        self.q_table[state][action_index] += self.learning_rate * (
            reward + self.discount_factor * best_next_action_value - self.q_table[state][action_index]
        )

        # Decay the exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    # def select_action(self, state):
    #     # Epsilon-greedy action selection
    #     # if random.random() < self.exploration_rate:
    #     if random.random() < .5:
    #         return random.choice(Actions.list())  # Explore: random action
    #     else:
    #         action_index = np.argmax(self.q_table[tuple(state)])  # Convert state to tuple
    #         return Actions.list()[action_index]  # Return action based on index    

    # def update(self, state, action, reward, next_state):
    #     # Update the Q-table using the Q-learning formula
    #     action_index = Actions.list().index(action)  # Get the index of the action
    #     best_next_action_value = np.max(self.q_table[tuple(next_state)])  # Convert next_state to tuple
        
    #     # Update the Q-value for the current state-action pair
    #     self.q_table[tuple(state)][action_index] += self.learning_rate * (
    #         reward + self.discount_factor * best_next_action_value - self.q_table[tuple(state)][action_index]
    #     )

    #     # Decay the exploration rate
    #     if self.exploration_rate > self.min_exploration_rate:
    #         self.exploration_rate *= self.exploration_decay

    def run(self, env, checkpoint="q_table_final.pkl", max_steps=100):
        # Load the checkpoint file
        self.load_q_table(checkpoint)

        state = env.reset()  # Reset the environment to the initial state
        done = False
        step = 0

        while not done and step < max_steps:
            action = self.select_action(state)  # Choose action based on the Q-table
            print(f'heere is the actin ')
            next_state, reward, done, _ = env_red.step(action)  # Perform action
            print(f' the reward is {reward}')
            # Move to the next state
            state = next_state
            step += 1

        print("Run completed.")


    # def update(self, state, action, reward, next_state):
    #     # Update the Q-table using the Q-learning formula
    #     action_index = Actions.list().index(action)  # Get the index of the action
    #     best_next_action_value = np.max(self.q_table[next_state])  # Best Q-value for next state
        
    #     # Update the Q-value for the current state-action pair
    #     self.q_table[state][action_index] += self.learning_rate * (
    #         reward + self.discount_factor * best_next_action_value - self.q_table[state][action_index]
    #     )

    #     # Decay the exploration rate
    #     if self.exploration_rate > self.min_exploration_rate:
    #         self.exploration_rate *= self.exploration_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as file:
            self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())), pickle.load(file))

# Training Loop
if __name__ == "__main__":
    env = env_red()  # Initialize the environment

    agent = AIAgent()
    total_episodes = 1000  # Total episodes to train
    max_steps_per_episode = 100  # Maximum steps per episode

    for episode in range(total_episodes):
        state = env.reset()  # Initialize state
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # Select an action using the agent's policy
            action = agent.select_action(state)
            
            # Take action in the environment and observe the outcome
            next_state, reward, done, _ = env.step(action)  # Replace with environment's step function

            # Update agent with new experience
            agent.update(state, action, reward, next_state)

            # Move to the next state
            state = next_state
            step += 1

        # Optional: Save the Q-table every 100 episodes
        if episode % 100 == 0:
            agent.save_q_table(f"q_table_checkpoint_{episode}.pkl")
            print(f"Checkpoint saved at episode {episode}")

    # Save final Q-table
    agent.save_q_table("q_table_final.pkl")
    print("Training completed. Final Q-table saved.")
   