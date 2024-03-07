import gymnasium as gym
import numpy as np


class RewardConverter(gym.Wrapper):
    def __init__(self, env, div_pos_rew=2048, default_neg=0, term_rew=-1):
        super().__init__(env)
        assert div_pos_rew > 0, "div_pos_rew must be greater than 0"
        self.div_pos_rew = div_pos_rew
        self.default_neg = default_neg
        self.term_rew = term_rew

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            if len(info["score_per_step"]) == 0:
                reward = self.default_neg
            else:
                reward = np.sum(np.array(info["score_per_step"])) / self.div_pos_rew
        else:
            reward = self.term_rew

        return obs, reward, terminated, truncated, info
