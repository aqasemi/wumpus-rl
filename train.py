import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from wumpus_env import WumpusWorldEnv

ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Grab', 'Climb']


def evaluate(model, difficulty, n=30):
    """Evaluate model win rate on given difficulty."""
    env = DummyVecEnv([lambda d=difficulty: WumpusWorldEnv(difficulty=d, max_steps=24)])
    wins = 0
    for _ in range(n):
        obs = env.reset()
        done = False
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break
        if info[0].get('win'):
            wins += 1
    env.close()
    return wins / n


def train_curriculum(save_path='models/ppo_wumpus'):
    """
    Train with curriculum learning.
    
    Stages:
    - Stage 1: Fixed gold at [3,1], no hazards
    - Stage 2: Random gold, no hazards
    - Stage 3: Random gold + wumpus
    - Stage 4: Full game with pits
    """
    print("=" * 50)
    print("WUMPUS WORLD - CURRICULUM TRAINING")
    print("=" * 50)
    
    os.makedirs('models', exist_ok=True)
    
    # Initialize on easiest level
    env = DummyVecEnv([lambda: WumpusWorldEnv(difficulty=0, max_steps=24)])
    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.02,
        verbose=0
    )
    
    stages = [
        (0, 5000, "Fixed gold"),
        (1, 20000, "Random gold"),
        (2, 30000, "+ Wumpus"),
        (3, 50000, "+ Pits (full game)"),
    ]
    
    for diff, steps, desc in stages:
        print(f"\nStage {diff + 1}: {desc} ({steps:,} steps)")
        
        env = DummyVecEnv([lambda d=diff: WumpusWorldEnv(difficulty=d, max_steps=24)])
        model.set_env(env)
        model.learn(steps, progress_bar=True)
        
        win_rate = evaluate(model, diff)
        print(f"  Win rate: {win_rate * 100:.0f}%")
        env.close()
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    for d in range(4):
        rate = evaluate(model, d, n=50)
        labels = ["Fixed gold", "Random gold", "+ Wumpus", "Full game"]
        print(f"  {labels[d]}: {rate * 100:.0f}%")
    
    model.save(save_path)
    print(f"\nModel saved: {save_path}")
    
    return model


def quick_train(difficulty=0, timesteps=10000):
    """Quick training on single difficulty."""
    print(f"\nTraining on difficulty {difficulty} for {timesteps:,} steps")
    
    env = DummyVecEnv([lambda d=difficulty: WumpusWorldEnv(difficulty=d, max_steps=24)])
    model = PPO('MlpPolicy', env, verbose=0, ent_coef=0.02)
    model.learn(timesteps, progress_bar=True)
    
    win_rate = evaluate(model, difficulty)
    print(f"Win rate: {win_rate * 100:.0f}%")
    
    env.close()
    return model


def watch_agent(model_path='models/ppo_wumpus', difficulty=1, n_episodes=3):
    """Watch trained agent play."""
    model = PPO.load(model_path)
    
    for ep in range(n_episodes):
        env = WumpusWorldEnv(difficulty=difficulty, render_mode='ansi', max_steps=24)
        obs, _ = env.reset()
        
        print(f"\n{'='*40}")
        print(f"Episode {ep + 1} (Difficulty {difficulty})")
        print("=" * 40)
        env.render()
        
        done = False
        total_reward = 0
        actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions.append(ACTION_NAMES[action])
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
            
            print(f"\nAction: {ACTION_NAMES[action]} | Reward: {reward:.1f}")
            env.render()
        
        result = "WIN!" if info.get('win') else "LOST"
        print(f"\n{result} | Total reward: {total_reward:.1f}")
        print(f"Actions: {' -> '.join(actions)}")
        env.close()


def record_gif(model_path='models/ppo_wumpus', difficulty=1, save_path='recordings'):
    """Record episode as GIF."""
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    
    os.makedirs(save_path, exist_ok=True)
    model = PPO.load(model_path)
    
    env = WumpusWorldEnv(difficulty=difficulty, render_mode='rgb_array', max_steps=24)
    obs, _ = env.reset()
    
    frames = [env.render()]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())
        done = term or trunc
    
    # Save GIF
    images = [Image.fromarray(f) for f in frames]
    gif_path = f'{save_path}/episode_diff{difficulty}.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
    
    result = "WIN" if info.get('win') else "LOST"
    print(f"Saved {gif_path} | {result} | {len(frames)} frames")
    
    env.close()
    return gif_path


def test_env():
    """Manual test of environment."""
    env = WumpusWorldEnv(difficulty=0, render_mode='ansi')
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Obs: {obs}")
    env.render()
    
    # Optimal: Right -> Grab -> Left -> Climb
    actions = [3, 4, 2, 5]
    
    total_reward = 0
    for a in actions:
        print(f"\nAction: {ACTION_NAMES[a]}")
        obs, reward, term, trunc, info = env.step(a)
        total_reward += reward
        print(f"Reward: {reward:.1f} | Total: {total_reward:.1f}")
        env.render()
        if term or trunc:
            print(f"Episode ended. Win: {info.get('win')}")
            break
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "test":
            test_env()
        elif cmd == "curriculum":
            train_curriculum()
        elif cmd == "watch":
            diff = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            watch_agent(difficulty=diff)
        elif cmd == "record":
            diff = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            record_gif(difficulty=diff)
        elif cmd.isdigit():
            quick_train(difficulty=int(cmd), timesteps=10000)
        else:
            print("Usage: python train.py [test|curriculum|watch|record|0|1|2|3]")
    else:
        train_curriculum()
