import gymnasium as gym
import numpy as np


class RewardConverter(gym.RewardWrapper):
    def __init__(self, env, goal=5.0, fail=-5, other=-0.001):
        super().__init__(env)
        self.goal = goal
        self.fail = fail
        self.other = other

    def reward(self, reward):
        if reward == 0:
            return self.other
        elif reward > 0:
            return self.goal
        else:
            return self.fail
