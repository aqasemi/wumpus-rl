# Wumpus World DQN

This project implements a simple DQN agent to solve the Wumpus World game using Gymnasium and PyTorch.

## Requirements

- Python 3.8+
- `gymnasium`
- `torch`
- `numpy`

## Installation

```bash
pip install gymnasium torch numpy
```

## Running the Code

### Train the Agent

To train the agent from scratch:

```bash
python main.py
```

This will train the agent for 1000 episodes and save the model to `wumpus_dqn.pth`.

### Visualize the Agent

To visualize a trained agent playing the game:

```bash
python main.py play
```

## Environment Details

- **Grid Size**: 5x5
- **Start Position**: Bottom-Left
- **Observation**: Full state (4 channels: Player, Pit, Wumpus, Gold)
- **Actions**: Up, Down, Left, Right
- **Rewards**:
  - Move: -1
  - Pit/Wumpus: -1000 (Game Over)
  - Gold: +100 (Game Continues)
