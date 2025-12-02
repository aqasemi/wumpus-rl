# Wumpus World RL - Diagnosis & Solution

## Problem Summary

Initial PPO training achieved 0% win rate after 80,000 steps. The agent either:
- Immediately climbed out (guaranteed small negative reward)
- Wandered randomly and died

## Root Causes Identified

1. **Complex action space**: Forward + TurnLeft + TurnRight requires learning orientation
2. **Sparse reward**: Win signal (+1000) never reached; agent only saw step costs and deaths
3. **Large observation**: 10 channels × 4×4 × 4 frames = 640 floats with sparse info
4. **No curriculum**: Agent thrown into full game complexity immediately

## Solution: Simplify Everything

### 1. Simplified Actions (6 → 6 but simpler)

**Before**: Forward, TurnLeft, TurnRight, Grab, Shoot, Climb
- Required learning orientation + movement coupling

**After**: Up, Down, Left, Right, Grab, Climb  
- Direct 4-directional movement, no orientation

### 2. Simplified Observation (640 → 8 floats)

```
[row, col, has_gold, glitter, can_win, gold_row, gold_col, danger_adjacent]
```

Key insight: Include only action-relevant information.

### 3. Dense Reward Shaping

| Event | Reward |
|-------|--------|
| Step | -1 |
| Move toward gold | +2 |
| Move toward start (with gold) | +5 |
| Grab gold | +20 |
| Climb with gold (WIN) | +100 |
| Climb without gold | -20 |
| Death | -100 |
| Bump wall | -3 |

### 4. Curriculum Learning

| Stage | Description | Steps | Win Rate |
|-------|-------------|-------|----------|
| 1 | Fixed gold at [3,1] | 5,000 | 100% |
| 2 | Random gold position | 20,000 | 100% |
| 3 | + Wumpus | 30,000 | 84% |
| 4 | + Pits (full game) | 50,000 | 78% |

## Results

**Total training**: ~105,000 steps (~3 minutes on CPU)

| Difficulty | Win Rate |
|------------|----------|
| Fixed gold | 100% |
| Random gold | 100% |
| + Wumpus | 84% |
| Full game | 78% |

## Key Lessons

1. **Start simple**: Get learning working on trivial version first
2. **Dense rewards**: Guide agent toward goal with shaping
3. **Action-relevant observations**: Only include info that affects decisions
4. **Curriculum**: Gradually increase complexity
5. **Iterate fast**: Short training runs to debug quickly

## Usage

```bash
# Test environment
python train.py test

# Full curriculum training
python train.py curriculum

# Watch trained agent
python train.py watch 2

# Record GIF
python train.py record 1
```
