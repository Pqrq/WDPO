import gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class KArmedBanditEnv(gym.Env):
    def __init__(self, k=10, seed=42, fixed_stds=False):
        super().__init__()
        self.k = k
        self.np_random = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1)  # Dummy observation space

        self.means = self.np_random.uniform(-1.0, 1.0, size=k)
        if fixed_stds:
            self.stds = np.ones(k)
        else:
            self.stds = self.np_random.normal(0.0, 1.0, size=k)


    def reset(self):
        return 0, {}  # returns dummy observation

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        mean = self.means[action]
        std = self.stds[action]
        reward = self.np_random.normal(mean, std)
        done = True  # Each step ends the episode
        info = {}
        return 0, reward, done, info  # returns dummy observation


class ContinuousBanditEnv(gym.Env):
    """
    A simple 1D continuous optimization landscape.
    Action: a float in range [-5, 5]
    Reward: -(action - target)^2
    (The agent must find the 'target' value)
    """
    def __init__(self, target=2.5):
        super().__init__()
        self.target = target
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) # Dummy obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        info = {}
        return np.array([0.0], dtype=np.float32), info

    def step(self, action):
        # Reward is negative squared distance to target (Parabola)
        obs = np.array([0.0], dtype=np.float32)
        reward = -((action - self.target) ** 2).item()
        terminated = True
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info