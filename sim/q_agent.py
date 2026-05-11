import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning agent.
    Q-table shape: [battery_levels+1, hours, n_actions] = [11, 24, 3]
    """

    def __init__(
        self,
        n_actions     = 3,
        battery_bins  = 11,
        hour_bins     = 24,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.05,
        epsilon_decay = 0.995
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = np.zeros((battery_bins, hour_bins, n_actions))

    def _discretize(self, obs):
        battery = int(np.clip(obs[0], 0, 10))
        hour    = int(np.clip(obs[1], 0, 23))
        return battery, hour

    def choose_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        b, h = self._discretize(obs)
        return int(np.argmax(self.q_table[b, h]))

    def update(self, obs, action, reward, next_obs, done):
        b,  h  = self._discretize(obs)
        nb, nh = self._discretize(next_obs)
        current_q = self.q_table[b, h, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[nb, nh])
        self.q_table[b, h, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_pkl(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Policy saved → {path}")

    def load_pkl(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Policy loaded ← {path}")

    def save_npy(self, path):
        np.save(path, self.q_table)
        print(f"Q-table saved → {path}")

    def load_npy(self, path):
        self.q_table = np.load(path)
        print(f"Q-table loaded ← {path}")
