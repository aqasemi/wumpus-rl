# Wumpus World - Model Analysis

## What We Implemented

### ✅ Partial Observability
- Agent only sees cells it has visited (fog of war)
- Breeze/stench indicators when adjacent to hazards
- Gold only visible when discovered

### ✅ Frame Stacking (4 frames)
- Gives implicit short-term memory
- Agent can see recent trajectory
- Lighter than LSTM, works on CPU

### ✅ Richer Observation (8 channels)
```
0: Agent position
1: Visited cells map
2: Breeze locations (danger hint)
3: Stench locations (danger hint)
4: Known safe cells
5: Known danger cells
6: Gold (when discovered)
7: Wall indicators
```

### ✅ Improved Rewards
```
+100 : Find gold (win)
+10  : Discover new cell
+5   : Brave exploration (explore despite nearby danger)
+2   : Move away from danger
-1   : Step cost
-3   : Hit wall
-5   : Excessive revisits
-50  : Death (pit/wumpus)
```

---

## Current Results

| Metric | Value |
|--------|-------|
| Win Rate | ~23% |
| Avg Exploration | 5-6 cells |
| Training Time | ~15 min (100k steps) |

---

## Why This Task is Hard

1. **Sparse Reward**: Gold is randomly placed, agent must explore to find it
2. **Stochastic Hazards**: Pits/wumpus positions vary each episode  
3. **Partial Info**: Agent can't see hazards until it's too late
4. **Exploration vs Safety**: Must balance exploring new cells vs avoiding death

---

## Further Improvements (Not Implemented)

### 1. Curriculum Learning
- Start with easier maps (fewer hazards)
- Gradually increase difficulty

### 2. Intrinsic Motivation
- Reward for reducing uncertainty
- Bonus for information gain

### 3. Better Architecture
- Attention mechanisms for spatial reasoning
- Separate exploration/exploitation policies

### 4. Planning Components
- Monte Carlo Tree Search
- World model for simulation

### 5. LSTM/GRU (if compute available)
- True sequence memory
- Better for long-horizon planning
