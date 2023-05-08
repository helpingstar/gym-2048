import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces

class Game2048(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, size=4, goal=2048):
        self.size = size
        self.window_width = 400
        self.window_height = 500

        self.temp_window_margin = 10

        self.board_margin = int((self.window_width - 2 * self.temp_window_margin) / (8 * size + 1))
        self.block_size = self.board_margin * 7

        self.board_size = self.board_margin * (size+1) + self.block_size * size
        self.window_margin = (self.window_width - self.board_size) // 2

        self.block_rect = (self.block_size, self.block_size)

        self.left_top_board = np.array([self.window_margin, self.window_height-self.window_margin-self.board_size])
        self.left_top_first_block = self.left_top_board + np.array([self.board_margin, self.board_margin])
        self.to_next_block = self.block_size + self.board_margin

        assert math.log2(goal).is_integer() and goal > 4, "Goal must be a power of 2 and bigger than 4."

        self.observation_space = spaces.Box(low=1, high=goal, shape=(size, size), dtype=np.float32)
        # 0: right, 1: up, 2: down, 3: left
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # rednering setting
        block_color = ["#eee4da", "#ede0c8", "#f2b179", "#f59563", "#f67c5f",
                       "#f65e3b", "#edcf72", "#edcc61", "#edc850", "#edc53f", "#edc22e"]
        self.game_color = {2 ** i: pygame.Color(color) for i, color in enumerate(block_color, 1)}
        self.game_color[0] = pygame.Color("#cdc1b4")
        self.game_color['background'] = pygame.Color("#faf8ef")
        self.game_color['board_background'] = pygame.Color("#bbada0")
        self.game_color['number_0'] = pygame.Color('#776e65')


        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.size, self.size), dtype=int)

        self.spawnblock()
        self.spawnblock()

        # total score
        self.score = 0
        # score per step
        self.score_per_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        return self.board.astype(np.float32)

    def _get_info(self):
        return {'score_per_step': self.score_per_step}

    def spawnblock(self):
        number = self.np_random.choice([2, 4], 1, p=(0.8, 0.2)).item()
        empty_list = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty_list.append((r, c))

        r, c = self.np_random.choice(empty_list, 1)[0]

        self.board[r][c] = number

    def step(self, action):
        # 0: left
        # 1: right
        # 2: up
        # 3: down

        self.score_per_step = 0

        is_changed = False

        if action < 2:
            for r in range(self.size):
                new_line = self._combiner(self.board[r, :], action)
                # Check to see if anything has changed due to the action.
                if not is_changed:
                    is_changed = np.any(self.board[r, :] != new_line)
                self.board[r, :] = new_line
        else:
            for c in range(self.size):
                new_line = self._combiner(self.board[:, c], action-2)
                # Check to see if anything has changed due to the action.
                if not is_changed:
                    is_changed = np.any(self.board[:, c] != new_line)
                self.board[:, c] = new_line

        self.score += self.score_per_step

        if is_changed:
            self.spawnblock()
            terminated = self._is_game_over()
            reward = terminated ^ 1
        else:
            reward = 0
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _combiner(self, line:np.ndarray, way:int) -> np.ndarray:
        """Combine identical blocks on a single line.

        Args:
            line (np.ndarray): Lines to be merged.
            way (int): left: 0, right: 1

        Returns:
            np.ndarray: combined line
        """

        # way must be 0 or 1
        assert way == 0 or way == 1

        new_line = np.zeros(self.size)
        is_combined = np.zeros(self.size)

        if way == 0:
            cur = 0
            for i in range(self.size):
                if line[i] != 0:
                    if cur == 0:
                        new_line[cur] = line[i]
                        cur += 1
                    else:
                        if new_line[cur-1] == line[i] and is_combined[cur-1] == 0:
                            new_line[cur-1] += new_line[cur-1]
                            # add score
                            # self.score_per_step must be int
                            self.score_per_step += int(new_line[cur-1])
                            is_combined[cur-1] = 1
                        else:
                            new_line[cur] = line[i]
                            cur += 1
        else:
            cur = self.size-1
            for i in range(self.size-1, -1, -1):
                if line[i] != 0:

                    if cur == self.size-1:
                        new_line[cur] = line[i]
                        cur -= 1
                    else:
                        if new_line[cur+1] == line[i] and is_combined[cur+1] == 0:
                            new_line[cur+1] += new_line[cur+1]
                            # add score
                            # self.score_per_step must be int
                            self.score_per_step += int(new_line[cur+1])
                            is_combined[cur+1] = 1
                        else:
                            new_line[cur] = line[i]
                            cur -= 1
        return new_line

    def _is_game_over(self):
        if np.any(self.board == 0):
            return False
        for r in range(self.size):
            for c in range(self.size-1):
                if self.board[r][c] == self.board[r][c+1]:
                    return False
        for c in range(self.size):
            for r in range(self.size-1):
                if self.board[r][c] == self.board[r+1][c]:
                    return False
        return True

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # (width, height)
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.font = pygame.font.Font(None, 50)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(self.game_color['background'])
        # Rect(left, top, width, height)
        pygame.draw.rect(
            canvas,
            self.game_color['board_background'],
            (self.window_margin, self.window_height-self.window_margin-self.board_size, self.board_size, self.board_size)
        )

        for r in range(self.size):
            for c in range(self.size):
                # render block
                left_top_this_block = (self.left_top_first_block + np.array([self.to_next_block * c, self.to_next_block * r]))
                pygame.draw.rect(
                    canvas,
                    self.game_color[self.board[r][c]],
                    (left_top_this_block, (self.block_rect))
                )
                # Empty parts do not output a number.
                if self.board[r][c] == 0:
                    continue
                # render text
                text = self.font.render(str(self.board[r][c]), True, self.game_color['number_0'])
                text_rect = text.get_rect(center=(left_top_this_block + np.array([self.block_size//2, self.block_size//2])))
                canvas.blit(text, text_rect)

        score = self.font.render(f'score: {self.score}', True, self.game_color['number_0'])
        canvas.blit(score, (self.window_width // 3 * 1, self.left_top_board[1] // 2))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
