import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WumpusWorldEnv(gym.Env):
    """
    Simplified Wumpus World - 4-directional movement.
    
    Actions: Up, Down, Left, Right, Grab, Climb
    No turning - agent can move in any direction directly.
    
    Difficulty levels:
    - 0: Gold at [3,1], no hazards
    - 1: Random gold, no hazards  
    - 2: Random gold + wumpus
    - 3: Full game with pits
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}

    # Actions (simplified - no Grab needed, auto-pickup)
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    CLIMB = 4
    
    # Direction vectors for movement
    DIR_VECTORS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Climb']

    def __init__(self, render_mode=None, difficulty=0, max_steps=40):
        super().__init__()
        self.grid_size = 4
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Climb
        
        # Observation: 14 floats
        # [row, col, has_gold, can_win,
        #  danger_up, danger_down, danger_left, danger_right,
        #  glitter_up, glitter_down, glitter_left, glitter_right,
        #  unvisited_up, unvisited_down, unvisited_left, unvisited_right]
        # Note: walls can be inferred from position (row=0 → wall up, col=3 → wall right, etc.)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.float32
        )
        
        self._reset_state()

    def _reset_state(self):
        self.agent_pos = [3, 0]  # Bottom-left
        self.wumpus_pos = None
        self.wumpus_alive = False
        self.gold_pos = None
        self.pits = []
        self.has_gold = False
        self.has_arrow = True
        self.game_over = False
        self.win = False
        self.current_step = 0
        self.visited = {(3, 0)}  # Track visited cells for exploration bonus

    def _adjacent(self, r, c):
        adj = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                adj.append((nr, nc))
        return adj

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        
        if self.difficulty == 0:
            # Easiest: gold one step right
            self.gold_pos = [3, 1]
            self.wumpus_pos = None
            self.wumpus_alive = False
            self.pits = []
        elif self.difficulty == 1:
            # Random gold, no hazards
            others = [(r,c) for r in range(4) for c in range(4) if (r,c) != (3,0)]
            idx = self.np_random.integers(0, len(others))
            self.gold_pos = list(others[idx])
            self.wumpus_pos = None
            self.wumpus_alive = False
            self.pits = []
        elif self.difficulty == 2:
            # Random gold + wumpus, no pits
            others = [(r,c) for r in range(4) for c in range(4) if (r,c) != (3,0)]
            idx = self.np_random.integers(0, len(others))
            self.gold_pos = list(others[idx])
            remaining = [c for c in others if list(c) != self.gold_pos]
            idx = self.np_random.integers(0, len(remaining))
            self.wumpus_pos = list(remaining[idx])
            self.wumpus_alive = True
            self.pits = []
        else:
            # Full difficulty
            others = [(r,c) for r in range(4) for c in range(4) if (r,c) != (3,0)]
            idx = self.np_random.integers(0, len(others))
            self.gold_pos = list(others[idx])
            remaining = [c for c in others if list(c) != self.gold_pos]
            idx = self.np_random.integers(0, len(remaining))
            self.wumpus_pos = list(remaining[idx])
            self.wumpus_alive = True
            self.pits = []
            for r, c in others:
                if abs(3-r) + abs(0-c) <= 1:
                    continue
                if [r,c] == self.gold_pos or [r,c] == self.wumpus_pos:
                    continue
                if self.np_random.random() < 0.15:
                    self.pits.append([r, c])
        
        return self._obs(), {}

    def _obs(self):
        r, c = self.agent_pos
        
        # Helper to check if cell is valid
        def in_bounds(nr, nc):
            return 0 <= nr < 4 and 0 <= nc < 4
        
        # Check for danger in each direction (pit or wumpus)
        def is_danger(nr, nc):
            if not in_bounds(nr, nc):
                return 0
            if [nr, nc] in self.pits:
                return 1
            if self.wumpus_pos and [nr, nc] == self.wumpus_pos and self.wumpus_alive:
                return 1
            return 0
        
        danger_up = is_danger(r - 1, c)
        danger_down = is_danger(r + 1, c)
        danger_left = is_danger(r, c - 1)
        danger_right = is_danger(r, c + 1)
        
        # Glitter in each direction (gold is adjacent - tells you WHERE!)
        def is_glitter(nr, nc):
            if not in_bounds(nr, nc):
                return 0
            if self.gold_pos and [nr, nc] == self.gold_pos and not self.has_gold:
                return 1
            return 0
        
        glitter_up = is_glitter(r - 1, c)
        glitter_down = is_glitter(r + 1, c)
        glitter_left = is_glitter(r, c - 1)
        glitter_right = is_glitter(r, c + 1)
        
        # Unvisited cell detection - helps agent explore systematically
        def is_unvisited(nr, nc):
            if not in_bounds(nr, nc):
                return 0
            return 0 if (nr, nc) in self.visited else 1
        
        unvisited_up = is_unvisited(r - 1, c)
        unvisited_down = is_unvisited(r + 1, c)
        unvisited_left = is_unvisited(r, c - 1)
        unvisited_right = is_unvisited(r, c + 1)
        
        # Can win = at start with gold
        at_start = (r == 3 and c == 0)
        can_win = 1 if (at_start and self.has_gold) else 0
        
        obs = np.array([
            r / 3.0,
            c / 3.0,
            1.0 if self.has_gold else 0.0,
            float(can_win),
            float(danger_up),
            float(danger_down),
            float(danger_left),
            float(danger_right),
            float(glitter_up),
            float(glitter_down),
            float(glitter_left),
            float(glitter_right),
            float(unvisited_up),
            float(unvisited_down),
            float(unvisited_left),
            float(unvisited_right),
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        self.current_step += 1
        reward = -1  # Step cost
        terminated = False
        truncated = False
        
        r, c = self.agent_pos
        
        if action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            dr, dc = self.DIR_VECTORS[action]
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < 4 and 0 <= nc < 4:
                self.agent_pos = [nr, nc]
                
                # Death check
                if self.wumpus_pos and [nr, nc] == self.wumpus_pos and self.wumpus_alive:
                    reward = -100
                    terminated = True
                    self.game_over = True
                elif [nr, nc] in self.pits:
                    reward = -100
                    terminated = True
                    self.game_over = True
                else:
                    # Exploration bonus - reward visiting new cells
                    if (nr, nc) not in self.visited:
                        self.visited.add((nr, nc))
                        reward += 5  # Exploration bonus
                    
                    # Auto-pickup gold when stepping on it
                    if self.gold_pos and [nr, nc] == self.gold_pos and not self.has_gold:
                        self.has_gold = True
                        self.gold_pos = None
                        reward += 50  # Found the gold!
                    
                    # When holding gold, guide toward start
                    if self.has_gold:
                        old_dist = abs(r - 3) + abs(c - 0)
                        new_dist = abs(nr - 3) + abs(nc - 0)
                        reward += (old_dist - new_dist) * 5
            else:
                reward -= 2  # Bump into wall
        
        elif action == self.CLIMB:
            if self.agent_pos == [3, 0]:
                if self.has_gold:
                    reward = 100
                    self.win = True
                else:
                    reward = -20
                terminated = True
                self.game_over = True
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 10
        
        if self.render_mode == "human":
            self.render()
        
        return self._obs(), reward, terminated, truncated, {"win": self.win, "has_gold": self.has_gold}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_visual()

    def _render_ansi(self):
        lines = [f"\nStep {self.current_step} | Gold:{'✓' if self.has_gold else '○'}"]
        lines.append("+" + "---+" * 4)
        
        for r in range(4):
            row = "|"
            for c in range(4):
                ch = []
                if [r,c] == self.agent_pos:
                    ch.append('A')
                if self.wumpus_pos and [r,c] == self.wumpus_pos:
                    ch.append('W' if self.wumpus_alive else 'w')
                if self.gold_pos and [r,c] == self.gold_pos:
                    ch.append('G')
                if [r,c] in self.pits:
                    ch.append('P')
                cell = ''.join(ch[:3]).center(3) if ch else ' . '
                row += cell + "|"
            lines.append(row)
            lines.append("+" + "---+" * 4)
        
        # Show percepts
        percepts = self._get_percepts()
        if percepts:
            lines.append("Percepts: " + " | ".join(percepts))
        
        out = '\n'.join(lines)
        print(out)
        return out

    def _get_percepts(self):
        """Get current percepts for display."""
        r, c = self.agent_pos
        percepts = []
        
        # Check for danger in each direction
        directions = [('^', r-1, c), ('v', r+1, c), ('<', r, c-1), ('>', r, c+1)]
        
        danger_dirs = []
        glitter_dirs = []
        
        for arrow, nr, nc in directions:
            if 0 <= nr < 4 and 0 <= nc < 4:
                # Danger (stench/breeze)
                if [nr, nc] in self.pits:
                    danger_dirs.append(arrow)
                if self.wumpus_pos and [nr, nc] == self.wumpus_pos and self.wumpus_alive:
                    danger_dirs.append(arrow)
                # Glitter
                if self.gold_pos and [nr, nc] == self.gold_pos and not self.has_gold:
                    glitter_dirs.append(arrow)
        
        if danger_dirs:
            percepts.append(f"DANGER {' '.join(danger_dirs)}")
        if glitter_dirs:
            percepts.append(f"GLITTER {' '.join(glitter_dirs)}")
        if self.has_gold:
            percepts.append("GOT GOLD")
        if r == 3 and c == 0 and self.has_gold:
            percepts.append("CAN WIN")
            
        return percepts

    def _render_visual(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.8, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        
        colors = {
            'cell': '#1f4068', 'agent': '#4ecca3', 'gold': '#ffd700',
            'wumpus': '#e94560', 'pit': '#0f0f0f', 'start': '#2a9d8f',
        }
        
        for r in range(4):
            for c in range(4):
                y = 3 - r
                color = colors['start'] if (r,c) == (3,0) else colors['cell']
                rect = patches.Rectangle((c, y), 1, 1, facecolor=color, edgecolor='#333', lw=2)
                ax.add_patch(rect)
                
                if [r,c] in self.pits:
                    ax.add_patch(patches.Circle((c+0.5, y+0.5), 0.3, facecolor=colors['pit']))
                    ax.text(c+0.5, y+0.5, 'P', ha='center', va='center', fontsize=12, color='#666')
                
                if self.wumpus_pos and [r,c] == self.wumpus_pos:
                    wc = colors['wumpus'] if self.wumpus_alive else '#553344'
                    ax.text(c+0.5, y+0.5, 'W', ha='center', va='center', fontsize=18, color=wc, weight='bold')
                
                if self.gold_pos and [r,c] == self.gold_pos:
                    ax.text(c+0.5, y+0.5, '★', ha='center', va='center', fontsize=20, color=colors['gold'])
                
                if [r,c] == self.agent_pos:
                    ax.add_patch(patches.Circle((c+0.5, y+0.5), 0.3, facecolor=colors['agent'], edgecolor='white', lw=2))
        
        # Status line
        status = f"Step:{self.current_step} Gold:{'✓' if self.has_gold else '○'}"
        ax.text(0, 4.3, status, fontsize=10, color='#aaa')
        
        # Win/Dead message
        if self.win:
            ax.text(2, 4.6, "WIN!", fontsize=16, ha='center', color=colors['gold'], weight='bold')
        elif self.game_over:
            ax.text(2, 4.6, "DEAD", fontsize=16, ha='center', color=colors['wumpus'], weight='bold')
        
        # Percepts display
        percepts = self._get_percepts()
        if percepts:
            percept_text = "  |  ".join(percepts)
            ax.text(2, -0.4, percept_text, fontsize=9, ha='center', color='#ddd', 
                   bbox=dict(boxstyle='round', facecolor='#2d2d44', edgecolor='#444', pad=0.3))
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
        plt.close(fig)
        return img

    def close(self):
        pass
