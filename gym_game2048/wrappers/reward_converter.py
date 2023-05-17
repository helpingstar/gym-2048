import gymnasium as gym
import numpy as np


class RewardConverter(gym.RewardWrapper):
    def __init__(self, env, terminated_reward=5.0, other=-0.001):
        super().__init__(env)
        self.other = other
        self.terminated_reward = terminated_reward

    def reward(self, reward):
        if reward == 0:
            return self.other
        else:
            return reward * self.terminated_reward
