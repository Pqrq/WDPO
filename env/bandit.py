import gym
import numpy as np
from gym import spaces


class KArmedBanditEnv(gym.Env):
    def __init__(self, k=10, seed=42, fixed_stds=False):
        super().__init__()
        self.k = k
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1)  # Dummy observation space

        self.means = self.np_random.uniform(-1.0, 1.0, size=k)
        if fixed_stds:
            self.stds = np.ones(k)
        else:
            self.stds = self.np_random.normal(0.0, 1.0, size=k)


    def reset(self):
        return 0  # returns dummy observation

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        mean = self.means[action]
        std = self.stds[action]
        reward = self.np_random.normal(mean, std)
        done = True  # Each step ends the episode
        info = {}
        return 0, reward, done, info  # returns dummy observation




