import pickle
import random


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)

    def __len__(self):
        return len(self.buffer)
