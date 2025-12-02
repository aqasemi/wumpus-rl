# Wumpus World RL Agent

A reinforcement learning agent that solves the Wumpus World problem using PPO (Proximal Policy Optimization) from Stable-Baselines3.

## Overview

The agent learns to navigate a 4×4 grid, find gold, and return to the exit while avoiding deadly hazards (Wumpus and pits). The key innovation is a carefully designed observation space with **directional percepts** that enable an MLP-based agent to learn systematic exploration without requiring memory (LSTM/GRU).

## Results

| Difficulty | Description | Win Rate |
|------------|-------------|----------|
| 0 | Fixed gold position | 100% |
| 1 | Random gold position | 100% |
| 2 | + Wumpus | ~95% |
| 3 | + Pits (full game) | ~80-85% |

## Quick Start

```bash
# Install dependencies
uv sync

# Test environment
uv run python train.py test

# Train with curriculum learning
uv run python train.py curriculum

# Watch trained agent play
uv run python train.py watch 3

# Record 10 episodes as GIFs
uv run python train.py record 3 10
```

## Environment

- **Grid**: 4×4 (16 cells)
- **Start**: Bottom-left `[3,0]`
- **Goal**: Find gold → Return to start → Climb out

### Actions (5)

| ID | Action | Description |
|----|--------|-------------|
| 0 | Up | Move up (row - 1) |
| 1 | Down | Move down (row + 1) |
| 2 | Left | Move left (col - 1) |
| 3 | Right | Move right (col + 1) |
| 4 | Climb | Exit cave (only at start) |

**Note**: Gold is automatically picked up when stepping on it.

### Observation Space (16 floats)

```
[row, col, has_gold, can_win,
 danger_up, danger_down, danger_left, danger_right,
 glitter_up, glitter_down, glitter_left, glitter_right,
 unvisited_up, unvisited_down, unvisited_left, unvisited_right]
```

Key design: **Directional percepts** tell the agent exactly which direction contains danger, gold hints (glitter), or unexplored cells.

### Difficulty Levels

| Level | Gold | Wumpus | Pits | Description |
|-------|------|--------|------|-------------|
| 0 | Fixed `[3,1]` | None | None | Tutorial: gold one step right |
| 1 | Random | None | None | Must explore to find gold |
| 2 | Random | Random | None | Must avoid wumpus |
| 3 | Random | Random | ~15% | Full game with all hazards |

## Training Approach

### Curriculum Learning

Training proceeds through stages of increasing difficulty:

| Stage | Difficulty | Steps | Description |
|-------|------------|-------|-------------|
| 1 | 0 | 10,000 | Fixed gold (learn basic mechanics) |
| 2 | 1 | 50,000 | Random gold (learn exploration) |
| 3 | 2 | 80,000 | Add wumpus (learn danger avoidance) |
| 4 | 3 | 100,000 | Add pits (full game) |

**Total**: ~240,000 steps (~5-10 minutes on CPU)

### Reward Structure

| Event | Reward |
|-------|--------|
| Each step | -1 |
| Visit new cell | +5 |
| Pick up gold | +50 |
| Move toward start (with gold) | +5 per step closer |
| Climb with gold (WIN) | +100 |
| Climb without gold | -20 |
| Bump into wall | -3 |
| Death (pit/wumpus) | -100 |
| Timeout (40 steps) | -10 |

## Files

```
wumpus/
├── wumpus_env.py      # Gymnasium environment
├── train.py           # Training & evaluation script
├── requirements.txt   # Dependencies
├── models/            # Saved model checkpoints
└── recordings/        # Episode GIFs
```

## Requirements

- Python 3.10+
- gymnasium
- stable-baselines3
- numpy
- matplotlib
- pillow
