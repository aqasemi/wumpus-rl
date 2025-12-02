import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from env.wumpus_env import WumpusEnv
from agents.dqn_agent import DQNAgent
from tqdm import tqdm
import json

print("All imports successful!")

def create_output_dirs():
    """Create necessary directories for saving models and results"""
    print("Creating output directories...")
    dirs = ['models', 'results', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("Directories created successfully!")

def train_custom_dqn(env, episodes=100, evaluate_every=20):
    """Train the custom DQN agent"""
    print("Training custom DQN...")
    # Calculate state dimension based on flattened observation space
    state_dim = (
        1 +  # grid_size
        2 +  # player_pos
        env.max_entities * 2 +  # wumpus_positions
        env.max_entities * 2 +  # pit_positions
        2 +  # gold_position
        1 +  # has_gold
        env.max_grid_size * env.max_grid_size  # visited_cells
    )
    
    agent = DQNAgent(state_dim=state_dim, action_dim=env.action_space.n)
    episode_rewards = []
    evaluation_scores = []
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            state = agent.preprocess_state(state)
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = agent.preprocess_state(next_state)
                
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")
            
            # Save model periodically
            if (episode + 1) % 20 == 0:
                agent.save(f"models/custom_dqn_episode_{episode + 1}.pth")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    
    # Save final model
    agent.save("models/custom_dqn_final.pth")
    print("Custom DQN training complete!")
    
    # Return results dictionary
    return {
        'episode_rewards': episode_rewards,
        'evaluation_scores': evaluation_scores,
        'mean_reward': np.mean(episode_rewards[-20:]),  # Mean of last 20 episodes
        'std_reward': np.std(episode_rewards[-20:])  # Std of last 20 episodes
    }

def train_stable_baselines(env, algo_name, total_timesteps=10000):
    """Train using Stable-Baselines3 algorithms"""
    print(f"\nTraining {algo_name}...")
    
    # Custom callback to track episode rewards
    class RewardCallback:
        def __init__(self):
            self.rewards = []
            self._current_reward = 0
        
        def __call__(self, locals_, globals_):
            # Get info from the locals dictionary
            info = locals_['infos'][0] if locals_['infos'] else None
            done = locals_['dones'][0]
            
            if info and 'episode' in info:
                episode_reward = info['episode']['r']
                self.rewards.append(episode_reward)
                print(f"Episode {len(self.rewards)}, Reward: {episode_reward:.2f}")
            
            return True
    
    reward_callback = RewardCallback()
    
    if algo_name == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=0)
    elif algo_name == "A2C":
        model = A2C("MultiInputPolicy", env, verbose=0)
    elif algo_name == "DQN":
        model = DQN("MultiInputPolicy", env, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=reward_callback)
    episode_rewards = reward_callback.rewards
    
    # Save the model
    model.save(f"models/{algo_name.lower()}_final")
    
    # Evaluate the model
    print(f"\nEvaluating {algo_name}...")
    eval_rewards = []
    n_eval_episodes = 10
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    print(f"{algo_name} evaluation complete!")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Return results dictionary
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'mean_reward': mean_reward,
        'std_reward': std_reward
    }

def plot_training_results(custom_results, sb_results, save_dir='plots'):
    """Plot and compare training results across different algorithms"""
    print("Plotting training results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot episode rewards
    plt.figure(figsize=(12, 8))
    plt.plot(custom_results['episode_rewards'], label='Custom DQN')
    for algo, results in sb_results.items():
        plt.plot(results['episode_rewards'], label=f'SB3 {algo}')
    plt.title('Training Rewards Across Algorithms')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'rewards_comparison_{timestamp}.png'))
    plt.close()
    
    # Plot evaluation metrics
    metrics = {
        'Custom DQN': custom_results['mean_reward'],
        **{f'SB3 {algo}': results['mean_reward'] for algo, results in sb_results.items()}
    }
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=list(metrics.values()), labels=list(metrics.keys()))
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Evaluation Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'performance_comparison_{timestamp}.png'))
    plt.close()

    print("Plotting complete!")

def main():
    print("Starting Wumpus World RL Training\n")
    create_output_dirs()
    
    # Initialize environment
    env = WumpusEnv()
    
    # Dictionary to store all results
    all_results = {}
    
    # Train Custom DQN
    custom_results = train_custom_dqn(env)
    all_results['custom_dqn'] = custom_results
    
    # Train Stable-Baselines3 algorithms
    algorithms = ["PPO", "A2C", "DQN"]
    for algo in algorithms:
        results = train_stable_baselines(env, algo)
        all_results[f'sb3_{algo.lower()}'] = results
    
    # Save results to JSON file
    results_file = 'results/training_results.json'
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\nTraining complete! Results saved to:", results_file)
    print("Use plot_results.py to visualize the results")

if __name__ == "__main__":
    main()
