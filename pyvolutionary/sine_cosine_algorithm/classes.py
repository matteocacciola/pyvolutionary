import numpy as np


class QTable:
    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        # Initialize the Q-table with zeros
        self.table = np.zeros((n_states, n_actions))
        # Define the ranges for r1 and r3
        self.r1_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        self.r3_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        # Define the ranges for density and distance
        self.density_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.distance_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.epsilon = 0.1

    def get_state(self, density, distance) -> float:
        density_range = next(i for i, r in enumerate(self.density_ranges) if density <= r[1])
        distance_range = next(i for i, r in enumerate(self.distance_ranges) if distance <= r[1])
        return density_range * 3 + distance_range

    def get_action(self, state) -> int:
        max_indices = np.argmax(self.table[state, :], keepdims=True)
        return np.random.choice(max_indices)

    def get_action_params(self, action) -> tuple:
        return self.r1_ranges[action // 3], self.r3_ranges[action % 3]

    def update(self, state, action, reward, alpha=0.1, gama=0.9) -> "QTable":
        self.table[state][action] += alpha * (reward + gama * np.max(self.table[state]) - self.table[state][action])
        return self
