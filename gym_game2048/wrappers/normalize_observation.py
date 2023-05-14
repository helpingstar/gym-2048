import gymnasium as gym
from gym.spaces import Box
import numpy as np

# Normalize the observation.
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, by_max=False, include_goal=True):
        """Initialize the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            by_max (bool, optional): True: divide by max of obs, False: divide by goal
            include_goal (bool, optional): True: divide by goal, False: divide by goal - 1, Because if reach goal, the game will be terminated.
        """
        super().__init__(env)
        self.by_max = by_max
        self.observation_space = Box(shape=(self.size, self.size), low=0, high=1, dtype=np.float32)
        self.include_goal = include_goal
        
    def observation(self, obs):
        if self.by_max:
            return obs / np.max(obs)
        else:
            if self.include_goal:
                return obs / self.goal
            else:
                return obs / (self.goal - 1)