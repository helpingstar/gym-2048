# gym-game2048

// TODO
1. 초기화할 때 블록의 위치를 고정하기 (오차를 줄이고 연산속도를 높임)
2. 큰 수에서의 블록 폰트

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
A reinforcement learning environment based on the game 2048.

The 2048 game is a single-player puzzle game where the objective is to combine tiles with the same number to create a tile with the value 2048. Tiles can be moved up, down, left, or right, and when two tiles with the same number touch, they merge into one tile with the combined value. The game ends when there are no more moves available, or when a tile with the value 2048 is created.

# Action Space
There are 4 discrete deterministic actions:
* 0 : Swipe left
* 1 : Swipe right
* 2 : Swipe up
* 3 : Swipe down

# Observation Space
The observation is a ndarray with shape (size, size, 1).

The elements correspond to the following:
* 0 : Blank cell
* 1~$\log_2 (\text{goal})$ : $2^n$

# Rewards
If you reach the goal, you get a reward of 1; if you don't reach the goal and the game ends, you get a reward of -1. In all other cases, you get a reward of 0.

# Arguments
* `size` : The size of the board. The board will be made to be the size of (size, size).
* `goal` : The number you want to reach. It should be entered as a power of two. The game ends when one of the numbers on the board becomes the `goal`.
