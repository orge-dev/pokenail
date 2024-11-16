import os
import pickle
import random
from collections import defaultdict

import numpy as np
import tqdm

from actions import Actions
from replay_buffer import ReplayBuffer


class AIAgent:
    def __init__(
        self,
        learning_rate=0.05,
        discount_factor=0.9,
        # 1.0 = random, not using q table
        # Setting this .8 (not very low) seems to make agent much prefer
        # walking upwards in starting building than 1.0 why... maybe not enough episodes/due to sparse reward (lookup reward shaping? or go straight to ppo?)
        exploration_rate=1.0,
        multiprocess_dict=None
    ):
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.multiprocess_dict = multiprocess_dict

    def select_action(self, state):
        """Selects the action with the highest Q-value from the Q-table for a given state."""
    
        possible_actions = Actions.list()

        # Exploit
        if np.random.random() >= self.exploration_rate:
            action_index = np.argmax(self.q_table[state])
            return possible_actions[action_index]

        # Explore
        tried_state_actions = self.multiprocess_dict
        if tried_state_actions is not None:
            if state not in tried_state_actions:
                tried_state_actions[state] = set()

            already = tried_state_actions[state]
            new_actions = [a for a in possible_actions if a not in already]
            if new_actions:
                choice = random.choice(new_actions)
                # For some reason, .add doesn't work, so we = like this
                tried_state_actions[state] = tried_state_actions[state] | {choice}
                return choice
    
        return random.choice(possible_actions)

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
        agent_id=None,
        replays_dir="replays",
        use_cumulative_rewards=False,  # TODO: Does this work?
        n_experiences=10000000,
        use_combined=False,  # TODO: Does this work?
        use_cumulative_reward_scaling=True,  # WIP
    ):
        """Train agent using stored replay expeiences"""

        os.makedirs("replays_combined", exist_ok=True)
        combined_path = "replays_combined/latest_combined.pkl"

        cumulative_rewards = []

        all_experiences = []
        if use_combined and os.path.exists(combined_path):
            print(f"Loading combined experiences from {combined_path}")
            with open(combined_path, "rb") as f:
                all_experiences = pickle.load(f)
        else:
            print("Loading individual replay files")
            for filename in tqdm.tqdm(
                list(os.listdir(replays_dir))[:1000]
            ):  # TODO: Revert 60
                if filename.endswith(".pkl"):
                    replay_buffer = ReplayBuffer()
                    replay_buffer.load(os.path.join(replays_dir, filename))
                    assert len(replay_buffer.buffer) > 1
                    last_experience = replay_buffer.buffer[-1]
                    cumulative_reward = last_experience["cumulative_reward"]
                    cumulative_rewards.append(cumulative_reward)
                    exps = [
                        (experience, False, cumulative_reward)
                        for experience in replay_buffer.buffer[:-1]
                    ] + [(replay_buffer.buffer[-1], True, cumulative_reward)]
                    all_experiences.extend(exps)

            # Save combined experiences
            # print("Saving combined experiences...")
            # with open(combined_path, "wb") as f:
            #     pickle.dump(all_experiences, f)
            # print(f"Saved combined experiences to {combined_path}")

        print(f"Collected {len(all_experiences)} experiences")
        print("Training from random sampling of experiences")
        samples = random.sample(
            all_experiences, min(n_experiences, len(all_experiences))
        )
        total_samples = len(samples)

        checkpoint_dir = None
        if agent_id is not None:
            checkpoint_dir = f"checkpoints_partial/{agent_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save initial state
            checkpoint_path = f"{checkpoint_dir}/checkpoint_0_percent.pkl"
            self.save_state(checkpoint_path, do_print=True)

        def get_reward_scaling(episode_reward, reward_thresholds):
            if episode_reward >= reward_thresholds["95p"]:
                return 200
            elif episode_reward >= reward_thresholds["90p"]:
                return 100
            elif episode_reward >= reward_thresholds["75p"]:
                return 50
            else:
                return 1

        reward_thresholds = {
            "95p": np.percentile(cumulative_rewards, 95),
            "90p": np.percentile(cumulative_rewards, 90),
            "75p": np.percentile(cumulative_rewards, 75),
        }
        print("reward thresholds", reward_thresholds)

        for sample_i, (
            experience,
            is_last_step_of_episode,
            episode_cumulative_reward,
        ) in enumerate(tqdm.tqdm(samples)):
            state = experience["state"]
            action = experience["action"]
            next_state = experience["next_state"]
            step_reward = experience["reward"]
            cumulative_reward = experience["cumulative_reward"]
            # cumulative reward is designed to only count on the last step of the episode
            if use_cumulative_rewards:
                if is_last_step_of_episode:
                    # 90th percentile of random movement 10000 episodes is 1000
                    # (makes sense? drunken walk sqrt(n)?)
                    # exclude runs that are less than 90%, then rerun experiment with new policy, and repeat?
                    if cumulative_reward < 1080:
                        cumulative_reward = 0
                    reward = cumulative_reward
                else:
                    reward = 0
            else:
                reward = step_reward

                if use_cumulative_reward_scaling:
                    scaling = get_reward_scaling(
                        episode_cumulative_reward, reward_thresholds
                    )
                    reward = reward * scaling

            if checkpoint_dir:
                # Save checkpoint every 10%
                progress = (sample_i + 1) / total_samples
                if progress * 100 % 10 == 0:  # At 10%, 20%, etc
                    checkpoint_path = (
                        f"{checkpoint_dir}/checkpoint_{int(progress*100)}_percent.pkl"
                    )
                    self.save_state(checkpoint_path, do_print=True)

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


def evaluate_training_progress(checkpoint_dir="checkpoints/training_progress"):
    """Run agents from all checkpoints simultaneously
    TODO: Display in grid or streamwrapper
    TODO: multiprocess/thread if possible
    """
    import math

    from env import EnvRed

    # Get all checkpoint files
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl")])
    n_agents = len(checkpoints)

    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(n_agents))

    # Initialize agents and environments
    agents = []
    envs = []

    print(f"Setting up {n_agents} agents in {grid_size}x{grid_size} grid...")

    for checkpoint in checkpoints:

        env = EnvRed(headless=False)
        # TODO: Lookup SDL position and/or pyboy for position/compositing

        # Create and load agent
        agent = AIAgent(exploration_rate=0.2)  # Low exploration to see learned behavior
        agent.load_state(os.path.join(checkpoint_dir, checkpoint))

        agents.append(agent)
        envs.append(env)

        print(f"Loaded checkpoint {checkpoint}")

    try:
        print("\nRunning episodes...")
        # Run all agents for same number of steps
        episode_length = 2000
        for _ in tqdm.tqdm(range(episode_length)):
            for env, agent in zip(envs, agents):
                state = env.previous_state
                action = agent.select_action(state)
                env.step(action)

    finally:
        print("Cleaning up...")
        for env in envs:
            env.close()
