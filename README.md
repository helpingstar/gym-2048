# gym-game2048

// TODO
1. README.md만들기
2. 초기화할 때 블록의 위치를 고정하기 (오차를 줄이고 연산속도를 높임)
3. 큰 수에서의 블록 폰트
4. userwarning, observation_space

# Example
```python
import gymnasium as gym
import gym_game2048
from gym_game2048.wrappers import LogObservation, NormalizeObservation

env = gym.make("gym_game2048/Game2048-v0", render_mode="human")
env = LogObservation(env)
env = NormalizeObservation(env, by_max=False, include_goal=False)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
       observation, info = env.reset()
env.close()
```


# Description

# Action Space

# Observation Space

# Rewards