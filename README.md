# Wumpus World RL Agent

A reinforcement learning agent that solves the classic Wumpus World problem using PPO (Proximal Policy Optimization) from Stable-Baselines3.

## Results

| Difficulty | Description | Win Rate |
|------------|-------------|----------|
| 0 | Fixed gold position | 100% |
| 1 | Random gold position | 100% |
| 2 | + Wumpus | ~70% |
| 3 | + Pits (full game) | ~65% |

## Environment

**Grid**: 4×4, agent starts at bottom-left `[3,0]`

**Actions** (6):
- `Up`, `Down`, `Left`, `Right` - Move one cell
- `Grab` - Pick up gold if on same cell
- `Climb` - Exit cave (only works at start cell)

**Objective**: Find the gold, grab it, return to start, and climb out.

**Hazards** (difficulty 2+):
- Wumpus: Kills agent on contact
- Pits: Instant death

## Quick Start

```bash
# Install dependencies
uv sync

# Test environment
uv run python train.py test

# Train with curriculum learning
uv run python train.py curriculum

# Watch trained agent
uv run python train.py watch 1

# Record GIF
uv run python train.py record 1
```

## Training Approach

### Curriculum Learning

The agent is trained in stages of increasing difficulty:

1. **Stage 1** (5k steps): Fixed gold at `[3,1]`, no hazards
2. **Stage 2** (20k steps): Random gold position, no hazards
3. **Stage 3** (30k steps): Add wumpus
4. **Stage 4** (50k steps): Add pits (full game)

### Key Design Decisions

1. **Simplified actions**: Direct 4-directional movement (no turning)
2. **Dense reward shaping**: Distance-to-goal rewards guide exploration
3. **Compact observation**: 8 floats with action-relevant info only
4. **Curriculum**: Gradually increase complexity

See `DIAGNOSIS.md` for detailed analysis of the learning challenges and solutions.

## Files

- `wumpus_env.py` - Gymnasium environment
- `train.py` - Training script with curriculum learning
- `DIAGNOSIS.md` - Analysis of learning challenges
- `recordings/` - Saved episode GIFs

## Requirements

- Python 3.10+
- gymnasium
- stable-baselines3
- numpy
- matplotlib
- pillow
