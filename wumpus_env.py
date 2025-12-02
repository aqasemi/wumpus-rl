import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class WumpusWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super(WumpusWorldEnv, self).__init__()

        self.grid_size = 5
        self.render_mode = render_mode

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        # Observation: 4 channels x 5 x 5
        # Channel 0: Player position
        # Channel 1: Pit positions
        # Channel 2: Wumpus position
        # Channel 3: Gold position
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 5, 5), dtype=np.float32)

        self.player_pos: list[float] = []
        self.wumpus_pos: list[list[float]] = []
        self.gold_pos: list[float] = []
        self.pits = []
        self.has_gold = False

        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.has_gold = False
        self.pits = []

        # Start at Bottom-Left (Row=4, Col=0)
        self.player_pos = [self.grid_size - 1, 0]

        # Define safe zones (Start + Adjacent)
        # Adjacent to (4,0) are (3,0) and (4,1)
        safe_cells = {(self.grid_size - 1, 0), (self.grid_size - 2, 0), (self.grid_size - 1, 1)}

        # Place Wumpus (1 or 2 Wumpus)
        num_wumpus = self.np_random.integers(1, 3)  # 1 or 2 Wumpus
        self.wumpus_pos = []
        for _ in range(num_wumpus):
            while True:
                r = self.np_random.integers(0, self.grid_size)
                c = self.np_random.integers(0, self.grid_size)
                if (r, c) not in safe_cells and (r, c) not in self.wumpus_pos:
                    self.wumpus_pos.append([r, c])
                    break

        # Place Gold (1 Gold)
        while True:
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            # Gold can be anywhere except start? Or truly random?
            # Usually not at start.
            if (r, c) != (self.grid_size - 1, 0):
                self.gold_pos = [r, c]
                break

        # Place Pits (Probabilistic or Fixed number? User said "simple", let's use probability 0.2 like standard)
        # But we must ensure safe cells are safe.
        self.pits = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in safe_cells:
                    continue
                if [r, c] in self.wumpus_pos: # Avoid placing pit on wumpus? Wumpus falls in pit? Let's keep distinct.
                    continue
                if [r, c] == self.gold_pos: # Pit on gold? Usually no.
                    continue

                if self.np_random.random() < 0.2:
                    self.pits.append([r, c])

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Player
        obs[0, self.player_pos[0], self.player_pos[1]] = 1.0

        # Channel 1: Pits
        for p in self.pits:
            obs[1, p[0], p[1]] = 1.0

        # Channel 2: Wumpus
        for w in self.wumpus_pos:
            obs[2, w[0], w[1]] = 1.0
    
        # Channel 3: Gold
        if not self.has_gold and self.gold_pos:
            obs[3, self.gold_pos[0], self.gold_pos[1]] = 1.0

        return obs

    def step(self, action):
        self.current_step += 1

        row, col = self.player_pos

        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0: # Up
            row -= 1
        elif action == 1: # Down
            row += 1
        elif action == 2: # Left
            col -= 1
        elif action == 3: # Right
            col += 1

        # Check boundary
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            # Hit wall
            reward = -1
            terminated = False
            truncated = False
            # Stay in same place
        else:
            self.player_pos = [row, col]
            reward = -1 # Move cost
            terminated = False
            truncated = False

            # Check interactions
            if [row, col] in self.wumpus_pos:
                reward += -1000
                terminated = True
            elif [row, col] in self.pits:
                reward += -1000
                terminated = True
            elif [row, col] == self.gold_pos and not self.has_gold:
                reward += 100
                self.has_gold = True
                # Remove gold from map effectively (it won't show in obs next step)
                self.gold_pos = None # Or just hide it
                # Game continues

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # ANSI render
        grid_str = ""
        for r in range(self.grid_size):
            line = ""
            for c in range(self.grid_size):
                cell_str = "."

                # Check static objects first
                if [r, c] in self.pits:
                    cell_str = "P"
                if [r, c] in self.wumpus_pos:
                    cell_str = "W"
                if [r, c] == self.gold_pos and not self.has_gold:
                    cell_str = "G"

                # Player overrides
                if [r, c] == self.player_pos:
                    if cell_str == "G":
                        cell_str = "+" # Player on Gold
                    elif cell_str == "P":
                        cell_str = "@" # Player in Pit
                    elif cell_str == "W":
                        cell_str = "X" # Player on Wumpus
                    else:
                        cell_str = "A" # Agent

                line += cell_str + " "
            grid_str += line + "\n"
        print(grid_str)
