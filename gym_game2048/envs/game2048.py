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
        self.goal = goal
        self.best_score = 0

        # number of episode
        self.n_episode = 0

        assert math.log2(goal).is_integer() and (
            2 < math.log2(goal) < 256
        ), "goal must be 0 or a power of 2 and 4 < goal < 2^256"

        # goal from the board's perspective
        self.board_goal = int(np.log2(goal))

        self.observation_space = spaces.Box(
            low=0, high=self.board_goal, shape=(1, size, size), dtype=np.uint8
        )

        # 0: left, 1: right, 2: up, 3: down
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.size, self.size), dtype=np.uint8)

        self.legal_moves = np.ones(4, np.uint8)
        self.is_legal = True

        self._spawn_block()
        self._spawn_block()

        # total score
        self.score = 0
        # score per step
        self.score_per_step = []
        # check_step
        self.n_step = 0

        self.n_episode += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        self._update_legal_moves()

        return observation, info

    def _get_obs(self):
        return np.expand_dims(self.board, axis=0)

    def _get_info(self):
        return {
            "score_per_step": self.score_per_step,
            "score": self.score,
            "max": np.max(self.board),
            "action_mask": self.legal_moves,
            "is_legal": self.is_legal,
        }

    def _check_if_same(self, way: int) -> bool:
        """Check if blocks with the same number are next to each other.

        Args:
            way (int): 0 : check row, 1: check column

        Returns:
            bool: Returns True if there are blocks with the same number side by side.
        """
        if way == 0:
            for r in range(self.size):
                for c in range(self.size - 1):
                    if self.board[r][c] == 0:
                        continue
                    if self.board[r][c] == self.board[r][c + 1]:
                        return True
        else:
            for c in range(self.size):
                for r in range(self.size - 1):
                    if self.board[r][c] == 0:
                        continue
                    if self.board[r][c] == self.board[r + 1][c]:
                        return True
        return False

    def _update_legal_moves(self):
        if self._check_if_same(0):
            self.legal_moves[0] = 1
            self.legal_moves[1] = 1
        else:
            self.legal_moves[0] = 0
            self.legal_moves[1] = 0

        if self._check_if_same(1):
            self.legal_moves[2] = 1
            self.legal_moves[3] = 1
        else:
            self.legal_moves[2] = 0
            self.legal_moves[3] = 0

        for i in range(4):
            if self.legal_moves[i] == 0 and self._check_legal(i):
                self.legal_moves[i] = 1
        return

    def _check_legal(self, way: int):
        assert 0 <= way <= 4, "Way: [0, 4]"
        if way == 0:  # 0: left
            checker = self.board[:, self.size - 1] != 0
            for i in range(self.size - 2, -1, -1):
                line = self.board[:, i] != 0
                if np.any((checker ^ line) & checker):
                    return True
                else:
                    checker |= line
        elif way == 1:  # 1: right
            checker = self.board[:, 0] != 0
            for i in range(1, self.size):
                line = self.board[:, i] != 0
                if np.any((checker ^ line) & checker):
                    return True
                else:
                    checker |= line
        elif way == 2:  # 2: up
            checker = self.board[self.size - 1, :] != 0
            for i in range(self.size - 2, -1, -1):
                line = self.board[i, :] != 0
                if np.any((checker ^ line) & checker):
                    return True
                else:
                    checker |= line

        else:  # 3: down
            checker = self.board[0, :] != 0
            for i in range(1, self.size):
                line = self.board[i, :] != 0
                if np.any((checker ^ line) & checker):
                    return True
                else:
                    checker |= line
        return False

    def _spawn_block(self):
        zero_pos = np.where(self.board == 0)
        idx = np.random.randint(len(zero_pos[0]))
        r, c = zero_pos[0][idx], zero_pos[1][idx]

        self.board[r][c] = self._random_1_2()

    def _random_1_2(self):
        return 1 if self.np_random.random() < 0.8 else 2

    def step(self, action):
        # 0: left
        # 1: right
        # 2: up
        # 3: down

        # increase n_step
        self.n_step += 1
        self.score_per_step = []

        # Check to see if anything has changed due to the action.
        is_changed = self.legal_moves[action] == 1

        if is_changed:
            self.is_legal = True
            if action < 2:
                for r in range(self.size):
                    new_line = self._combiner(self.board[r, :], action)
                    self.board[r, :] = new_line
            else:
                for c in range(self.size):
                    new_line = self._combiner(self.board[:, c], action - 2)
                    self.board[:, c] = new_line
            self.score += sum(self.score_per_step)

            # If the goal is reached, set the reward to 1.
            if self._is_reach_goal():
                reward = 1
                terminated = True
            else:
                self._spawn_block()
                terminated = self._is_game_over()
                if terminated:
                    self._update_best_score()
                    reward = -1
                else:
                    reward = 0

        else:
            self.is_legal = False
            terminated = False
            reward = 0

        self._update_legal_moves()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _combiner(self, line: np.ndarray, way: int) -> np.ndarray:
        """Combine identical blocks on a single line.

        Args:
            line (np.ndarray): Lines to be merged.
            way (int): left: 0, right: 1

        Returns:
            np.ndarray: combined line
        """

        # way must be 0 or 1
        assert way == 0 or way == 1

        new_line = np.zeros(self.size, dtype=np.int32)
        is_combined = np.zeros(self.size, dtype=np.int32)

        if way == 0:
            cur = 0
            for i in range(self.size):
                if line[i] != 0:
                    if cur == 0:
                        new_line[cur] = line[i]
                        cur += 1
                    else:
                        if new_line[cur - 1] == line[i] and is_combined[cur - 1] == 0:
                            new_line[cur - 1] += 1
                            # add score
                            # self.score_per_step must be int
                            self.score_per_step.append(2 ** new_line[cur - 1])
                            is_combined[cur - 1] = 1
                        else:
                            new_line[cur] = line[i]
                            cur += 1
        else:
            cur = self.size - 1
            for i in range(self.size - 1, -1, -1):
                if line[i] != 0:
                    if cur == self.size - 1:
                        new_line[cur] = line[i]
                        cur -= 1
                    else:
                        if new_line[cur + 1] == line[i] and is_combined[cur + 1] == 0:
                            new_line[cur + 1] += 1
                            # add score
                            # self.score_per_step must be int
                            self.score_per_step.append(2 ** new_line[cur + 1])
                            is_combined[cur + 1] = 1
                        else:
                            new_line[cur] = line[i]
                            cur -= 1
        return new_line

    def _is_game_over(self):
        if np.any(self.board == 0):
            return False
        if self._check_if_same(0) or self._check_if_same(1):
            return False
        return True

    def _update_best_score(self):
        self.best_score = max(self.best_score, self.score)

    def _is_reach_goal(self):
        return np.any(self.board == self.board_goal)

    def _render_block(self, r, c, canvas: pygame.Surface):
        number = self.board[r][c]
        pygame.draw.rect(
            canvas,
            self.block_color[min(11, number)],
            (
                (self.block_x_pos[c], self.block_y_pos[r]),
                (self.block_size, self.block_size),
            ),
        )
        # Empty parts do not output a number.
        if self.board[r][c] == 0:
            return

        # render number
        if number < 7:
            size = self.block_font_size[0]
        elif number < 10:
            size = self.block_font_size[1]
        elif number < 13:
            size = self.block_font_size[2]
        elif number < 20:
            size = self.block_font_size[3]
        else:
            size = self.block_font_size[2]
        font = pygame.font.Font(None, size)

        num_str = str(2 ** self.board[r][c]) if number < 20 else f"2^{number}"

        color = self.block_font_color[0] if number < 3 else self.block_font_color[1]
        text = font.render(num_str, True, color)
        text_rect = text.get_rect(
            center=(
                (
                    self.block_x_pos[c] + self.block_size // 2,
                    self.block_y_pos[r] + self.block_size // 2,
                )
            )
        )
        canvas.blit(text, text_rect)

    def _render_info(self, canvas):
        info_font = pygame.font.Font(None, 35)
        score = info_font.render(f"score: {self.score}", True, (119, 110, 101))
        best_score = info_font.render(f"best: {self.best_score}", True, (119, 110, 101))

        canvas.blit(score, (15, 25))
        canvas.blit(best_score, (15, 65))

    def render(self):
        pygame.font.init()
        if self.window is None:
            pygame.init()

            # rendering : Size
            win_mg = 10

            self.board_size = self.window_width - 2 * win_mg
            self.block_size = int(self.board_size / (8 * self.size + 1) * 7)

            self.block_x_pos = np.zeros(self.size)
            self.block_y_pos = np.zeros(self.size)

            self.left_top_board = (
                win_mg,
                self.window_height - win_mg - self.board_size,
            )
            gap = self.board_size / (1 + 8 * self.size)

            for i in range(self.size):
                self.block_x_pos[i] = int(self.left_top_board[0] + (8 * i + 1) * gap)
                self.block_y_pos[i] = int(self.left_top_board[1] + (8 * i + 1) * gap)

            # rendering: Block Color
            self.block_color = [
                (205, 193, 180),
                (238, 228, 218),
                (237, 224, 200),
                (242, 177, 121),
                (245, 149, 99),
                (246, 124, 95),
                (246, 94, 59),
                (237, 207, 114),
                (237, 204, 97),
                (237, 200, 80),
                (237, 197, 63),
                (237, 194, 46),
            ]
            self.game_color = {}
            self.game_color["background"] = pygame.Color("#faf8ef")
            self.game_color["board_background"] = pygame.Color("#bbada0")
            self.block_font_color = [(119, 110, 101), (249, 246, 242)]

            # rendering: Block Font Size
            self.block_font_size = [
                int(self.block_size * rate) for rate in [0.7, 0.6, 0.5, 0.4]
            ]

            if self.render_mode == "human":
                pygame.display.init()
                # (width, height)
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
            else:
                self.window = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # rendering: Info

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(self.game_color["background"])
        # Rect(left, top, width, height)
        pygame.draw.rect(
            canvas,
            self.game_color["board_background"],
            (self.left_top_board, (self.board_size, self.board_size)),
        )

        for r in range(self.size):
            for c in range(self.size):
                # render block
                self._render_block(r, c, canvas)

        self._render_info(canvas)

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
