import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Normalize the observation.
class Normalize2048(gym.ObservationWrapper):
    def __init__(self, env):

        super().__init__(env)
        self.observation_space = spaces.Box(shape=(self.size, self.size, 1), low=0, high=1, dtype=np.float32)

    def observation(self, obs):
        return obs / self.board_goal
