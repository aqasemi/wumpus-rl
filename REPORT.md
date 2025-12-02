# Wumpus World RL - Environment & Agent Report

## Overview

This project implements a reinforcement learning agent for the classic Wumpus World problem using PPO (Proximal Policy Optimization) from Stable-Baselines3.

---

## Environment Specification

### Grid Layout

```
+---+---+---+---+
| . | . | . | . |  ← Row 0 (top)
+---+---+---+---+
| . | . | . | . |  ← Row 1
+---+---+---+---+
| . | . | . | . |  ← Row 2
+---+---+---+---+
| A | G | . | . |  ← Row 3 (bottom) | A=Agent start, G=Gold (difficulty 0)
+---+---+---+---+
  ↑   ↑   ↑   ↑
 C0  C1  C2  C3
```

- **Size**: 4×4 grid (16 cells)
- **Start position**: `[3, 0]` (bottom-left corner)
- **Coordinate system**: `[row, column]` where row 0 is top

### Actions (6 total)

| ID | Action | Description |
|----|--------|-------------|
| 0 | Up | Move one cell up (row - 1) |
| 1 | Down | Move one cell down (row + 1) |
| 2 | Left | Move one cell left (col - 1) |
| 3 | Right | Move one cell right (col + 1) |
| 4 | Grab | Pick up gold if on same cell |
| 5 | Climb | Exit cave (only works at start `[3,0]`) |

### Observation Space (11 floats)

```python
[row, col, has_gold, glitter, can_win, gold_row, gold_col, 
 danger_up, danger_down, danger_left, danger_right]
```

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | row | [0, 1] | Agent row (normalized by 3.0) |
| 1 | col | [0, 1] | Agent column (normalized by 3.0) |
| 2 | has_gold | {0, 1} | 1 if agent has picked up gold |
| 3 | glitter | {0, 1} | 1 if gold is on current cell |
| 4 | can_win | {0, 1} | 1 if at start AND has gold |
| 5 | gold_row | [0, 1] | Gold row position (normalized) |
| 6 | gold_col | [0, 1] | Gold column position (normalized) |
| 7 | danger_up | {0, 1} | 1 if pit/wumpus is UP |
| 8 | danger_down | {0, 1} | 1 if pit/wumpus is DOWN |
| 9 | danger_left | {0, 1} | 1 if pit/wumpus is LEFT |
| 10 | danger_right | {0, 1} | 1 if pit/wumpus is RIGHT |

**Key insight**: Directional danger signals are critical. A single "danger nearby" bit doesn't tell the agent which direction to avoid.

### Difficulty Levels

| Level | Gold | Wumpus | Pits | Description |
|-------|------|--------|------|-------------|
| 0 | Fixed `[3,1]` | None | None | Easiest: gold is one step right |
| 1 | Random | None | None | Must navigate to unknown location |
| 2 | Random | Random | None | Must avoid wumpus |
| 3 | Random | Random | 15% each cell* | Full game with all hazards |

*Pits excluded from cells within Manhattan distance 1 of start.

### Reward Structure

| Event | Reward | Notes |
|-------|--------|-------|
| Each step | -1 | Time penalty |
| Move toward gold | +2 | When not holding gold |
| Move toward start | +5 | When holding gold |
| Bump wall | -3 | Invalid move |
| Grab gold | +20 | Only when on gold cell |
| Climb with gold | +100 | **WIN condition** |
| Climb without gold | -20 | Penalty for giving up |
| Death (pit/wumpus) | -100 | Episode terminates |
| Timeout (24 steps) | -10 | Episode truncates |

---

## Agent Architecture

### Algorithm: PPO (Proximal Policy Optimization)

```python
PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    ent_coef=0.02,  # Entropy for exploration
    verbose=0
)
```

### Policy Network: MlpPolicy

- **Input**: 8-dimensional observation vector
- **Hidden layers**: [64, 64] (default SB3 architecture)
- **Output**: 6 action logits + value estimate

### Training Strategy: Curriculum Learning

Training proceeds through 4 stages of increasing difficulty:

| Stage | Difficulty | Steps | Cumulative | Expected Behavior |
|-------|------------|-------|------------|-------------------|
| 1 | 0 (Fixed gold) | 5,000 | 5,000 | Learn: Right → Grab → Left → Climb |
| 2 | 1 (Random gold) | 20,000 | 25,000 | Generalize navigation |
| 3 | 2 (+ Wumpus) | 30,000 | 55,000 | Learn danger avoidance |
| 4 | 3 (+ Pits) | 50,000 | 105,000 | Handle stochastic hazards |

**Total training**: ~105,000 steps (~2-3 minutes on CPU)

---

## Expected Performance

Based on training with directional danger signals:

| Difficulty | Win Rate | Notes |
|------------|----------|-------|
| 0 (Fixed gold) | 100% | Deterministic optimal path |
| 1 (Random gold) | 100% | Perfect navigation |
| 2 (+ Wumpus) | 100% | Agent avoids wumpus reliably |
| 3 (Full game) | ~88% | Fails only on impossible layouts |

### Failure Modes

1. **Gold on wumpus cell**: Agent cannot retrieve gold without dying
2. **Gold surrounded by hazards**: No safe path exists
3. **Unlucky pit generation**: May block all paths to gold

---

## Key Design Decisions

### Why Simplified Actions?

The original textbook Wumpus World uses `Forward`, `TurnLeft`, `TurnRight` which requires the agent to learn:
- Orientation state (N/E/S/W)
- Action coupling (turn + move = 2 steps to change direction)

**Our approach**: Direct 4-directional movement removes this complexity, allowing the agent to focus on the navigation problem.

### Why Dense Rewards?

Sparse rewards (+1000 for winning) fail because:
- Agent rarely reaches the goal during random exploration
- No learning signal for intermediate progress

**Our approach**: Shape rewards guide the agent toward the goal:
- Move toward gold → positive reward
- Move toward start (with gold) → larger positive reward

### Why Compact Observations?

Large observations (e.g., full grid state as image) require:
- More training data to generalize
- Larger networks to process
- Longer training time

**Our approach**: 8 floats containing only action-relevant information:
- Where am I? (row, col)
- Do I have the goal? (has_gold)
- Is goal here? (glitter)
- Can I win now? (can_win)
- Where is the goal? (gold_row, gold_col)
- Is it dangerous nearby? (danger)

---

## File Structure

```
wumpus/
├── wumpus_env.py      # Gymnasium environment implementation
├── train.py           # Training script with curriculum learning
├── main.py            # Interactive play mode (pygame)
├── REPORT.md          # This document
├── DIAGNOSIS.md       # Historical problem analysis
├── README.md          # Quick start guide
├── requirements.txt   # Dependencies
├── models/            # Saved model checkpoints
│   └── ppo_wumpus.zip
├── recordings/        # Episode GIFs
└── assets/            # Visualization sprites
```

---

## Usage Commands

```bash
# Test environment manually
uv run python train.py test

# Full curriculum training
uv run python train.py curriculum

# Quick training on single difficulty
uv run python train.py 0  # or 1, 2, 3

# Watch trained agent play
uv run python train.py watch 1

# Record episode as GIF
uv run python train.py record 1
```

---

## Visualization

The environment supports three render modes:

1. **ansi**: ASCII art in terminal
2. **rgb_array**: Matplotlib figure as numpy array (for GIFs)
3. **human**: Interactive pygame window (via main.py)

### Color Scheme (rgb_array mode)

| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark navy | `#1a1a2e` |
| Cell | Navy blue | `#1f4068` |
| Start cell | Teal | `#2a9d8f` |
| Agent | Mint green | `#4ecca3` |
| Gold | Gold | `#ffd700` |
| Wumpus | Red | `#e94560` |
| Pit | Black | `#0f0f0f` |

---

*Generated: December 2024*
