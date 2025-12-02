import os
import warnings

# Suppress gym deprecation warning (we're using gymnasium)
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import json

from wumpus_env import WumpusWorldEnv


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    """
    def __init__(self, eval_env, eval_freq=1000, log_dir="logs", verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.wins = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Track episode stats from info
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        rewards = self.locals.get('rewards', [0])
        
        self.current_episode_reward += rewards[0]
        self.current_episode_length += 1
        
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Track wins
            if infos and len(infos) > 0 and 'win' in infos[0]:
                self.wins.append(1 if infos[0]['win'] else 0)
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=10, deterministic=True
            )
            self.eval_rewards.append(mean_reward)
            self.eval_timesteps.append(self.num_timesteps)
            
            if self.verbose:
                win_rate = np.mean(self.wins[-50:]) * 100 if self.wins else 0
                print(f"\nStep {self.num_timesteps}: "
                      f"Eval Reward = {mean_reward:.2f} ± {std_reward:.2f}, "
                      f"Win Rate (last 50) = {win_rate:.1f}%")
        
        return True

    def _on_training_end(self) -> None:
        # Save training history (convert numpy types to Python types)
        def convert_to_python(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_to_python(x) for x in obj]
            return obj
        
        history = {
            'episode_rewards': convert_to_python(self.episode_rewards),
            'episode_lengths': convert_to_python(self.episode_lengths),
            'eval_rewards': convert_to_python(self.eval_rewards),
            'eval_timesteps': convert_to_python(self.eval_timesteps),
            'wins': convert_to_python(self.wins)
        }
        
        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)


def create_env(render_mode=None):
    """Create a monitored Wumpus environment."""
    env = WumpusWorldEnv(render_mode=render_mode)
    return env


def plot_training_results(history, save_path='plots'):
    """Generate training progress plots."""
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode Rewards with moving average
    ax1 = axes[0, 0]
    rewards = history['episode_rewards']
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax1.plot(range(49, len(rewards)), moving_avg, color='blue', linewidth=2, label='Moving Avg (50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation Rewards
    ax2 = axes[0, 1]
    if history['eval_rewards']:
        ax2.plot(history['eval_timesteps'], history['eval_rewards'], 
                color='green', marker='o', linewidth=2)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Evaluation Performance')
        ax2.grid(True, alpha=0.3)
    
    # Win Rate Over Time
    ax3 = axes[1, 0]
    wins = history.get('wins', [])
    if wins:
        window = 100
        if len(wins) >= window:
            win_rate = [np.mean(wins[max(0, i-window):i+1]) * 100 
                       for i in range(len(wins))]
            ax3.plot(win_rate, color='orange', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title(f'Win Rate (Rolling {window} episodes)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    # Episode Lengths
    ax4 = axes[1, 1]
    lengths = history['episode_lengths']
    ax4.plot(lengths, alpha=0.3, color='purple', label='Raw')
    if len(lengths) > 50:
        moving_avg = np.convolve(lengths, np.ones(50)/50, mode='valid')
        ax4.plot(range(49, len(lengths)), moving_avg, color='purple', linewidth=2, label='Moving Avg (50)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.set_title('Episode Lengths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_progress_{timestamp}.png'), dpi=150)
    plt.close()
    print(f"Training plots saved to {save_path}/")


def train_ppo(total_timesteps=50000, n_envs=4, save_path='models'):
    """
    Train PPO agent on Wumpus World.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_path: Directory to save models
    """
    print("=" * 60)
    print("Wumpus World - PPO Training")
    print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create vectorized environment for training
    print(f"\nCreating {n_envs} parallel training environments...")
    env = make_vec_env(create_env, n_envs=n_envs)
    
    # Create evaluation environment
    eval_env = create_env()
    
    # PPO hyperparameters tuned for this environment
    print("\nInitializing PPO agent...")
    
    # Custom network architecture for better learning
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Larger networks
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        policy="MlpPolicy",  # MLP for small grid observation (4x5x5=100 features)
        env=env,
        learning_rate=3e-4,
        n_steps=256,  # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="./logs/tensorboard/",
        device="auto"
    )
    
    print(f"Policy architecture: {model.policy}")
    print(f"Using device: {model.device}")
    
    # Setup callback
    callback = TrainingCallback(
        eval_env=eval_env,
        eval_freq=2000,
        log_dir='logs',
        verbose=1
    )
    
    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print("-" * 60)
    print("\nTraining complete!")
    
    # Save the final model
    model_path = os.path.join(save_path, "ppo_wumpus_final")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Final evaluation
    print("\nFinal Evaluation (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Count wins in final evaluation
    wins = 0
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get('win', False):
            wins += 1
    print(f"Win Rate: {wins/20*100:.1f}%")
    
    # Plot results
    history = {
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'eval_rewards': callback.eval_rewards,
        'eval_timesteps': callback.eval_timesteps,
        'wins': callback.wins
    }
    plot_training_results(history)
    
    env.close()
    eval_env.close()
    
    return model, history


def visualize_agent(model_path='models/ppo_wumpus_final', episodes=5, delay=0.5):
    """
    Visualize a trained agent playing the game.
    
    Args:
        model_path: Path to saved model
        episodes: Number of episodes to visualize
        delay: Delay between steps (seconds)
    """
    import time
    
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = create_env(render_mode='human')
    
    print(f"\nVisualizing {episodes} episodes...")
    print("Close the pygame window to stop.\n")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"Episode {ep + 1}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            time.sleep(delay)
        
        result = "WON!" if info.get('win', False) else "Lost"
        print(f"  Result: {result}, Reward: {total_reward:.0f}, Steps: {step}")
    
    env.close()


def demo_random_agent(episodes=3):
    """Quick demo with random actions to test the environment."""
    print("\nRunning random agent demo...")
    
    env = create_env(render_mode='human')
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            import time
            time.sleep(0.3)
        
        print(f"Episode {ep + 1}: Reward = {total_reward:.0f}, Steps = {step}")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            # Visualize trained agent
            model_path = sys.argv[2] if len(sys.argv) > 2 else "models/ppo_wumpus_final"
            visualize_agent(model_path=model_path)
        elif sys.argv[1] == "demo":
            # Random agent demo
            demo_random_agent()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python train.py [play|demo]")
    else:
        # Train the agent
        model, history = train_ppo(
            total_timesteps=100000,  # 100k steps for decent learning
            n_envs=4
        )
        
        # Ask to visualize
        try:
            response = input("\nVisualize trained agent? (y/n): ")
            if response.lower() == 'y':
                visualize_agent()
        except EOFError:
            pass
