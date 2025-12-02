import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os


class WumpusWorldEnv(gym.Env):
    """
    Wumpus World Environment with visual rendering.
    
    A classic AI problem where an agent navigates a cave to find gold
    while avoiding pits and the deadly Wumpus.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None, grid_size=5):
        super(WumpusWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 100  # Pixel size for each cell
        
        # Pygame initialization (deferred until first render)
        self.window = None
        self.clock = None
        self.images = None

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        self.action_names = ["Up", "Down", "Left", "Right"]

        # Observation: 4 channels x grid_size x grid_size
        # Channel 0: Player position
        # Channel 1: Pit positions (observed/known)
        # Channel 2: Wumpus position (observed/known)
        # Channel 3: Gold position (observed/known)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(4, self.grid_size, self.grid_size), 
            dtype=np.float32
        )

        # Game state
        self.player_pos = []
        self.wumpus_pos = []
        self.gold_pos = []
        self.pits = []
        self.has_gold = False
        self.game_over = False
        self.win = False

        self.max_steps = 100
        self.current_step = 0
        
        # Track visited cells for reward shaping
        self.visited = set()

    def _load_images(self):
        """Load images for rendering."""
        import pygame
        
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        
        self.images = {}
        image_files = {
            'agent': 'agent.png',
            'wumpus': 'wumpus.png',
            'gold': 'gold.png',
            'pit': 'pit.png',
            'stench': 'stench.png',
            'breeze': 'breeze.png',
            'empty': 'empty.jpg'
        }
        
        for name, filename in image_files.items():
            path = os.path.join(assets_dir, filename)
            if os.path.exists(path):
                img = pygame.image.load(path)
                self.images[name] = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                # Create placeholder surface
                surf = pygame.Surface((self.cell_size, self.cell_size))
                surf.fill((100, 100, 100))
                self.images[name] = surf

    def _get_adjacent_cells(self, row, col):
        """Get valid adjacent cells."""
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                adjacent.append((nr, nc))
        return adjacent

    def _has_stench(self, row, col):
        """Check if cell has stench (adjacent to Wumpus)."""
        for w in self.wumpus_pos:
            if [row, col] in [list(adj) for adj in self._get_adjacent_cells(w[0], w[1])]:
                return True
        return False

    def _has_breeze(self, row, col):
        """Check if cell has breeze (adjacent to pit)."""
        for p in self.pits:
            if [row, col] in [list(adj) for adj in self._get_adjacent_cells(p[0], p[1])]:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.has_gold = False
        self.game_over = False
        self.win = False
        self.pits = []
        self.visited = set()

        # Start at Bottom-Left (Row=grid_size-1, Col=0)
        self.player_pos = [self.grid_size - 1, 0]
        self.visited.add(tuple(self.player_pos))

        # Define safe zones (Start + Adjacent)
        safe_cells = {
            (self.grid_size - 1, 0), 
            (self.grid_size - 2, 0), 
            (self.grid_size - 1, 1)
        }

        # Place Wumpus (1 or 2)
        num_wumpus = self.np_random.integers(1, 3)
        self.wumpus_pos = []
        for _ in range(num_wumpus):
            for attempt in range(100):  # Prevent infinite loop
                r = self.np_random.integers(0, self.grid_size)
                c = self.np_random.integers(0, self.grid_size)
                if (r, c) not in safe_cells and [r, c] not in self.wumpus_pos:
                    self.wumpus_pos.append([r, c])
                    break

        # Place Gold (1 Gold, not at start)
        for attempt in range(100):
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            if (r, c) != (self.grid_size - 1, 0):
                self.gold_pos = [r, c]
                break

        # Place Pits with 0.15 probability (slightly reduced for playability)
        self.pits = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in safe_cells:
                    continue
                if [r, c] in self.wumpus_pos:
                    continue
                if [r, c] == self.gold_pos:
                    continue
                if self.np_random.random() < 0.15:
                    self.pits.append([r, c])

        return self._get_obs(), {}

    def _get_obs(self):
        """Get the observation as a 4-channel grid."""
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
        old_pos = tuple(self.player_pos)

        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:  # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        # Check boundary
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            # Hit wall - stay in place, small penalty
            reward = -5
            terminated = False
            truncated = False
        else:
            self.player_pos = [row, col]
            reward = -1  # Move cost
            terminated = False
            truncated = False

            # Exploration bonus for visiting new cells
            if tuple(self.player_pos) not in self.visited:
                reward += 2  # Small bonus for exploration
                self.visited.add(tuple(self.player_pos))

            # Check interactions
            if [row, col] in self.wumpus_pos:
                reward = -100  # Death by Wumpus
                terminated = True
                self.game_over = True
            elif [row, col] in self.pits:
                reward = -100  # Death by pit
                terminated = True
                self.game_over = True
            elif self.gold_pos and [row, col] == self.gold_pos and not self.has_gold:
                reward = 100  # Found gold!
                self.has_gold = True
                self.win = True
                self.gold_pos = None
                terminated = True  # Game ends when gold is found

        if self.current_step >= self.max_steps:
            truncated = True

        # Auto-render in human mode
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {
            "has_gold": self.has_gold,
            "game_over": self.game_over,
            "win": self.win
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_pygame()
        return None

    def _render_ansi(self):
        """Text-based rendering."""
        grid_str = "\n" + "=" * (self.grid_size * 2 + 1) + "\n"
        for r in range(self.grid_size):
            line = "|"
            for c in range(self.grid_size):
                cell_str = "."

                # Check static objects
                if [r, c] in self.pits:
                    cell_str = "P"
                if [r, c] in self.wumpus_pos:
                    cell_str = "W"
                if self.gold_pos and [r, c] == self.gold_pos and not self.has_gold:
                    cell_str = "G"

                # Player overrides
                if [r, c] == self.player_pos:
                    if cell_str == "G":
                        cell_str = "+"  # Player on Gold
                    elif cell_str == "P":
                        cell_str = "@"  # Player in Pit (dead)
                    elif cell_str == "W":
                        cell_str = "X"  # Player on Wumpus (dead)
                    else:
                        cell_str = "A"  # Agent

                line += cell_str + "|"
            grid_str += line + "\n"
        grid_str += "=" * (self.grid_size * 2 + 1)
        print(grid_str)
        return grid_str

    def _render_pygame(self):
        """Visual rendering using pygame."""
        import pygame
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 60)
            )
            pygame.display.set_caption("Wumpus World")
            self.clock = pygame.time.Clock()
            self._load_images()

        # Create surface
        canvas = pygame.Surface(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 60)
        )
        canvas.fill((40, 40, 40))  # Dark background

        # Draw grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = c * self.cell_size
                y = r * self.cell_size

                # Draw empty cell background
                canvas.blit(self.images['empty'], (x, y))

                # Draw breeze indicator (semi-transparent overlay)
                if self._has_breeze(r, c):
                    breeze_img = self.images['breeze'].copy()
                    breeze_img.set_alpha(150)
                    canvas.blit(breeze_img, (x, y))

                # Draw stench indicator (semi-transparent overlay)
                if self._has_stench(r, c):
                    stench_img = self.images['stench'].copy()
                    stench_img.set_alpha(150)
                    canvas.blit(stench_img, (x, y))

                # Draw pit
                if [r, c] in self.pits:
                    canvas.blit(self.images['pit'], (x, y))

                # Draw wumpus
                if [r, c] in self.wumpus_pos:
                    canvas.blit(self.images['wumpus'], (x, y))

                # Draw gold
                if self.gold_pos and [r, c] == self.gold_pos and not self.has_gold:
                    canvas.blit(self.images['gold'], (x, y))

                # Draw player
                if [r, c] == self.player_pos:
                    canvas.blit(self.images['agent'], (x, y))

                # Draw grid lines
                pygame.draw.rect(canvas, (80, 80, 80), 
                               (x, y, self.cell_size, self.cell_size), 2)

        # Draw status bar
        font = pygame.font.Font(None, 28)
        status_y = self.grid_size * self.cell_size + 10
        
        # Step counter
        step_text = font.render(f"Step: {self.current_step}/{self.max_steps}", True, (200, 200, 200))
        canvas.blit(step_text, (10, status_y))
        
        # Gold status
        gold_color = (255, 215, 0) if self.has_gold else (150, 150, 150)
        gold_text = font.render(f"Gold: {'✓' if self.has_gold else '○'}", True, gold_color)
        canvas.blit(gold_text, (200, status_y))
        
        # Game status
        if self.game_over and not self.win:
            status_text = font.render("GAME OVER!", True, (255, 50, 50))
            canvas.blit(status_text, (350, status_y))
        elif self.win:
            status_text = font.render("VICTORY!", True, (50, 255, 50))
            canvas.blit(status_text, (350, status_y))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        """Clean up pygame resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
