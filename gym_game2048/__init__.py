from gymnasium.envs.registration import register

register(
     id="gym_game2048/Game2048-v0",
     entry_point="gym_game2048.envs:Game2048",
)
