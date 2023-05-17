import gymnasium as gym
import numpy as np


class RewardByScore(gym.Wrapper):
    def __init__(self, env, log=True):
        super().__init__(env)
        self.log = log

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        if self.log:
            reward = np.sum(np.log2(np.array(info['score_per_step']))) if len(info['score_per_step']) != 0 else 0
        else:
            reward = np.sum(np.array(info['score_per_step'])) if len(info['score_per_step']) != 0 else 0
        return obs, reward, terminated, truncated, info