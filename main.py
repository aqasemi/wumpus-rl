import gymnasium as gym
import wumpus_env
import numpy as np
import time
from dqn_agent import DQNAgent

def train():
    env = wumpus_env.WumpusWorldEnv()

    # State shape is (4, 5, 5)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    agent = DQNAgent(state_shape, action_dim, lr=0.001, epsilon_decay=0.995)

    num_episodes = 1000
    target_update_freq = 10

    print("Starting Training...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    print("Training Complete.")
    agent.save("wumpus_dqn.pth")
    return agent

def visualize(agent=None):
    if agent is None:
        env = wumpus_env.WumpusWorldEnv()
        state_shape = env.observation_space.shape
        action_dim = env.action_space.n
        agent = DQNAgent(state_shape, action_dim)
        try:
            agent.load("wumpus_dqn.pth")
            print("Loaded model.")
        except:
            print("No model found, using random agent.")

    env = wumpus_env.WumpusWorldEnv(render_mode='ansi')
    state, _ = env.reset()
    env.render()

    done = False
    truncated = False
    total_reward = 0
    step = 0

    # Turn off exploration
    agent.epsilon = 0.0

    print("\nVisualizing...")
    while not (done or truncated):
        # time.sleep(0.5) # Removed for batch run
        action = agent.select_action(state)

        action_name = ["Up", "Down", "Left", "Right"][action]
        print(f"Step {step}: Action {action_name}")

        state, reward, done, truncated, _ = env.step(action)
        env.render()
        total_reward += reward
        step += 1

        if done:
            if reward <= -1000:
                print("Game Over: Died!")
            elif reward >= 100:
                 pass

    if truncated:
        print("Game Over: Max steps reached.")

    print(f"Final Score: {total_reward}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        visualize()
    else:
        trained_agent = train()
        visualize(trained_agent)
