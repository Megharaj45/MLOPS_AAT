import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnergyGridEnv(gym.Env):
    """
    Custom RL Environment for Renewable Energy Grid Scheduling.
    SDG 7  – Affordable and Clean Energy
    SDG 9  – Industry, Innovation and Infrastructure
    SDG 13 – Climate Action

    State  : (battery_level, hour_of_day)
    Actions: 0=Use Renewable, 1=Use Battery, 2=Use Grid
    Reward : +10 renewable, +3 battery, -5 grid
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(EnergyGridEnv, self).__init__()
        self.max_battery = 10
        self.max_hours   = 24

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low  = np.array([0, 0],                                 dtype=np.float32),
            high = np.array([self.max_battery, self.max_hours - 1], dtype=np.float32),
            dtype = np.float32
        )
        self.reset()

    def _get_solar_wind(self, hour):
        solar = max(0, np.sin(np.pi * (hour - 6) / 12))
        wind  = 0.3 + 0.2 * np.sin(np.pi * hour / 12)
        return min(1.0, solar + wind)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour          = 0
        self.battery_level = np.random.randint(3, 8)
        self.total_carbon  = 0
        self.energy_log    = []
        obs = np.array([self.battery_level, self.hour], dtype=np.float32)
        return obs, {}

    def step(self, action):
        renewable_avail = self._get_solar_wind(self.hour)

        if action == 0:
            if renewable_avail >= 0.3:
                reward = 10
                source = "renewable"
                self.battery_level = min(
                    self.max_battery,
                    self.battery_level + int(renewable_avail * 2)
                )
            else:
                reward = -3
                source = "grid_fallback"
                self.total_carbon += 1

        elif action == 1:
            if self.battery_level > 0:
                reward = 3
                source = "battery"
                self.battery_level -= 1
            else:
                reward = -5
                source = "grid_fallback"
                self.total_carbon += 1

        else:
            reward = -5
            source = "grid"
            self.total_carbon += 1

        self.energy_log.append(source)
        self.hour += 1
        terminated = self.hour >= self.max_hours
        obs = np.array([self.battery_level, self.hour % self.max_hours], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self):
        print(f"Hour {self.hour:02d} | Battery: {self.battery_level} | Carbon: {self.total_carbon}")
