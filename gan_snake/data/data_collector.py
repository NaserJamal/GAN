import numpy as np

class DataCollector:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.states = []
        self.actions = []

    def add_sample(self, state, action):
        if len(self.states) >= self.max_samples:
            self.states.pop(0)
            self.actions.pop(0)
        
        self.states.append(state)
        self.actions.append(action)

    def get_data(self):
        return np.array(self.states), np.array(self.actions)

    def save_data(self, filename):
        np.savez(filename, states=np.array(self.states), actions=np.array(self.actions))

    @staticmethod
    def load_data(filename):
        data = np.load(filename)
        return data['states'], data['actions']