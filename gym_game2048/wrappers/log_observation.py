import gymnasium as gym
from gym.spaces import Box
import numpy as np

# Reduce the variance by applying the log2 operation to observations that are squared to the power of 2.
class LogObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(self.size, self.size), low=0, high=np.log2(self.goal), dtype=np.float32)
        self.goal = int(np.log2(self.goal))

    def observation(self, obs):
        return np.log2(np.where(obs == 0, 1, obs))