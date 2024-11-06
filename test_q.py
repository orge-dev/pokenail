
import unittest
import numpy as np
from collections import defaultdict
from actions import Actions

class SimpleGridEnv:
    def __init__(self):
        self.grid = np.zeros((5, 5))
        self.grid[2, 3] = -10  # bigger penalty for obstacle
        self.grid[2, 4] = 10   # bigger reward for goal
        self.pos = [2, 2]     # start in center
        
    def reset(self):
        self.pos = [2, 2]
        return {"position": tuple(self.pos)}
        
    def step(self, action):
        new_pos = self.pos.copy()
        if action == "UP": new_pos[0] -= 1
        elif action == "DOWN": new_pos[0] += 1
        elif action == "LEFT": new_pos[1] -= 1
        elif action == "RIGHT": new_pos[1] += 1
        
        # Check bounds
        if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5:
            self.pos = new_pos
            
        # Get reward and done status
        reward = self.grid[self.pos[0], self.pos[1]]
        done = reward != 0  # Episode ends on obstacle or goal
        
        # Small penalty for each step to encourage efficiency
        if not done:
            reward = -0.1
            
        return {"position": tuple(self.pos)}, reward, done, {}

class TestQLearning(unittest.TestCase):
    def test_simple_navigation(self):
        env = SimpleGridEnv()
        q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        learning_rate = 0.2  # Increased learning rate
        discount_factor = 0.9
        
        # Train for more episodes
        for episode in range(200):  # Doubled episodes
            state = env.reset()
            done = False
            
            while not done:
                state_tuple = tuple(state.items())
                
                # Slower epsilon decay
                if np.random.random() < max(0.1, 1.0 - episode/150):  # Slower decay
                    action = np.random.choice(Actions.list())
                else:
                    action = Actions.list()[np.argmax(q_table[state_tuple])]
                
                next_state, reward, done, _ = env.step(action)
                next_state_tuple = tuple(next_state.items())
                
                # Q-learning update
                best_next_value = np.max(q_table[next_state_tuple])
                q_table[state_tuple][Actions.list().index(action)] += learning_rate * (
                    reward + discount_factor * best_next_value - 
                    q_table[state_tuple][Actions.list().index(action)]
                )
                
                state = next_state
        
        # Test that agent learned optimal path
        state = env.reset()
        path = []
        done = False
        
        while not done and len(path) < 10:
            state_tuple = tuple(state.items())
            action = Actions.list()[np.argmax(q_table[state_tuple])]
            path.append(action)
            state, reward, done, _ = env.step(action)
            
        # Optimal path should be UP, RIGHT, RIGHT, DOWN
        optimal_actions = ['UP', 'RIGHT', 'RIGHT', 'DOWN']
        self.assertTrue(
            any(action in optimal_actions for action in path[:4]),
            f"Agent didn't learn optimal path. Got: {path}"
        )
        print(f"{path=}")
        self.assertLess(len(path), 6, "Agent took too long to reach goal")

if __name__ == '__main__':
    # Create a test suite with our test case
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQLearning)
    print("suite", suite)

    runner = unittest.TextTestRunner()
    
    # Run the test multiple times
    for i in range(1000):
        print(f"\nTest run {i+1}/10")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestQLearning)
        result = runner.run(suite)
