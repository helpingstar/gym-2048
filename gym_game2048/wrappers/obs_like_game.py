import gymnasium as gym
from gym.spaces import Box
import numpy as np

# Reduce the variance by applying the log2 operation to observations that are squared to the power of 2.
class ObsLikeGame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(self.size, self.size, 1), low=0, high=self.goal, dtype=np.float32)

    def observation(self, obs):
        return 2 ** obs
