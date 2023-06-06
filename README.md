The code that trained this environment can be found below.

[rl-application-gym-game2048](https://github.com/helpingstar/rl-application-gym-game2048)

# gym-game2048

https://github.com/helpingstar/gym-game2048/assets/54899900/b276a92b-bf1c-4e0f-aaf8-0236d38a5e7e

# Install
```bash
git clone https://github.com/helpingstar/gym-game2048.git
cd gym-game2048
pip install -r requirements.txt
pip install -e .
```


# Example
```python
import gymnasium as gym
import gym_game2048

env = gym.make("gym_game2048/Game2048-v0", render_mode="human")

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
The observation is a ndarray with shape (1, size, size).

The elements correspond to the following:
* 0 : Blank cell
* 1~ $\log_{2} (\text{goal})$ : $2^n$

# Rewards
If you reach the goal, you get a reward of 1; if you don't reach the goal and the game ends, you get a reward of -1. In all other cases, you get a reward of 0.

If you want to change the reward policy, use Wrappers.
# Arguments
* `size` : The size of the board. The board will be made to be the size of (size, size).
* `goal` : The number you want to reach. It should be entered as a power of two. The game ends when one of the numbers on the board becomes the `goal`.

# Wrappers
