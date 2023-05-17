import gymnasium as gym
import numpy as np

# Normalize the observation.
class PreprocessForTensor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))
