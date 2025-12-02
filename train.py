import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from wumpus_env import WumpusWorldEnv

ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Climb']


class MetricsCallback(BaseCallback):
    """
    Custom callback to track training metrics for visualization.
    """
    def __init__(self, eval_freq=1000, eval_episodes=20, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        
        # Episode-level metrics (from training)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        
        # Evaluation metrics (periodic)
        self.eval_timesteps = []
        self.eval_win_rates = []
        self.eval_mean_rewards = []
        self.eval_mean_lengths = []
        
        # Current episode tracking
        self._current_rewards = 0
        self._current_length = 0
        
        # Stage tracking
        self.stage_boundaries = []
        self.current_difficulty = 0
    
    def set_difficulty(self, difficulty):
        """Called when curriculum stage changes."""
        self.current_difficulty = difficulty
        self.stage_boundaries.append({
            'timestep': self.num_timesteps if hasattr(self, 'num_timesteps') else 0,
            'difficulty': difficulty
        })
    
    def _on_step(self) -> bool:
        # Track current episode
        self._current_rewards += self.locals['rewards'][0]
        self._current_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            self.episode_rewards.append(self._current_rewards)
            self.episode_lengths.append(self._current_length)
            self.episode_wins.append(1 if info.get('win', False) else 0)
            
            self._current_rewards = 0
            self._current_length = 0
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        """Run evaluation episodes."""
        env = WumpusWorldEnv(difficulty=self.current_difficulty, max_steps=40)
        
        rewards = []
        lengths = []
        wins = []
        
        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(action))
                ep_reward += reward
                ep_length += 1
                done = term or trunc
            
            rewards.append(ep_reward)
            lengths.append(ep_length)
            wins.append(1 if info.get('win', False) else 0)
        
        env.close()
        
        self.eval_timesteps.append(self.num_timesteps)
        self.eval_win_rates.append(np.mean(wins))
        self.eval_mean_rewards.append(np.mean(rewards))
        self.eval_mean_lengths.append(np.mean(lengths))
    
    def get_metrics(self):
        """Return all collected metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_wins': self.episode_wins,
            'eval_timesteps': self.eval_timesteps,
            'eval_win_rates': self.eval_win_rates,
            'eval_mean_rewards': self.eval_mean_rewards,
            'eval_mean_lengths': self.eval_mean_lengths,
            'stage_boundaries': self.stage_boundaries,
        }


def plot_training_metrics(metrics, save_dir='plots'):
    """Generate and save training visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme
    colors = {
        'primary': '#4ecca3',
        'secondary': '#e94560',
        'accent': '#ffd700',
        'bg': '#1a1a2e',
        'grid': '#333355',
        'text': '#e0e0e0',
        'stages': ['#4ecca3', '#45b7d1', '#f9ca24', '#e94560']
    }
    
    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(colors['bg'])
        ax.set_title(title, color=colors['text'], fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, color=colors['text'], fontsize=11)
        ax.set_ylabel(ylabel, color=colors['text'], fontsize=11)
        ax.tick_params(colors=colors['text'])
        ax.grid(True, alpha=0.3, color=colors['grid'])
        for spine in ax.spines.values():
            spine.set_color(colors['grid'])
    
    def add_stage_markers(ax, boundaries, ymin, ymax):
        """Add vertical lines for curriculum stage transitions."""
        stage_names = ['Fixed Gold', 'Random Gold', '+ Wumpus', '+ Pits']
        for i, boundary in enumerate(boundaries):
            ts = boundary['timestep']
            diff = boundary['difficulty']
            if ts > 0:
                ax.axvline(x=ts, color=colors['stages'][diff], linestyle='--', alpha=0.7, linewidth=1.5)
            # Add label at top
            if diff < len(stage_names):
                ax.text(ts + 1000, ymax * 0.95, stage_names[diff], 
                       color=colors['stages'][diff], fontsize=8, alpha=0.9)
    
    # Smooth function for noisy data
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # =========================================================================
    # PLOT 1: Training Overview (2x2 grid)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(colors['bg'])
    fig.suptitle('Wumpus World RL - Training Metrics', 
                 color=colors['accent'], fontsize=16, fontweight='bold', y=0.98)
    
    # 1a. Episode Rewards
    ax = axes[0, 0]
    style_ax(ax, 'Episode Rewards', 'Episode', 'Total Reward')
    rewards = metrics['episode_rewards']
    ax.plot(rewards, alpha=0.2, color=colors['primary'], linewidth=0.5)
    if len(rewards) > 50:
        smoothed = smooth(rewards)
        ax.plot(range(25, 25 + len(smoothed)), smoothed, color=colors['primary'], linewidth=2, label='Smoothed (50 ep)')
    ax.axhline(y=0, color=colors['text'], linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'], labelcolor=colors['text'])
    
    # 1b. Episode Lengths
    ax = axes[0, 1]
    style_ax(ax, 'Episode Lengths', 'Episode', 'Steps')
    lengths = metrics['episode_lengths']
    ax.plot(lengths, alpha=0.2, color=colors['secondary'], linewidth=0.5)
    if len(lengths) > 50:
        smoothed = smooth(lengths)
        ax.plot(range(25, 25 + len(smoothed)), smoothed, color=colors['secondary'], linewidth=2, label='Smoothed (50 ep)')
    ax.axhline(y=40, color=colors['accent'], linestyle='--', alpha=0.5, label='Max steps')
    ax.legend(loc='upper right', facecolor=colors['bg'], edgecolor=colors['grid'], labelcolor=colors['text'])
    
    # 1c. Win Rate Over Training
    ax = axes[1, 0]
    style_ax(ax, 'Win Rate (Evaluation)', 'Timesteps', 'Win Rate')
    ts = metrics['eval_timesteps']
    wr = metrics['eval_win_rates']
    ax.plot(ts, wr, color=colors['accent'], linewidth=2, marker='o', markersize=4)
    ax.fill_between(ts, 0, wr, color=colors['accent'], alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color=colors['text'], linestyle='--', alpha=0.3)
    if metrics['stage_boundaries']:
        add_stage_markers(ax, metrics['stage_boundaries'], 0, 1.05)
    
    # 1d. Rolling Win Rate (from training episodes)
    ax = axes[1, 1]
    style_ax(ax, 'Rolling Win Rate (Training)', 'Episode', 'Win Rate (100 ep window)')
    wins = metrics['episode_wins']
    if len(wins) > 100:
        rolling_wr = np.convolve(wins, np.ones(100)/100, mode='valid')
        ax.plot(range(50, 50 + len(rolling_wr)), rolling_wr, color=colors['primary'], linewidth=2)
        ax.fill_between(range(50, 50 + len(rolling_wr)), 0, rolling_wr, color=colors['primary'], alpha=0.2)
    else:
        ax.plot(wins, color=colors['primary'], linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color=colors['text'], linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_dir}/training_overview.png', dpi=150, facecolor=colors['bg'])
    plt.close()
    print(f"  Saved: {save_dir}/training_overview.png")
    
    # =========================================================================
    # PLOT 2: Evaluation Metrics Detail
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(colors['bg'])
    fig.suptitle('Evaluation Metrics During Training', 
                 color=colors['accent'], fontsize=14, fontweight='bold', y=1.02)
    
    # 2a. Win Rate
    ax = axes[0]
    style_ax(ax, 'Win Rate', 'Timesteps', 'Win Rate')
    ax.plot(ts, wr, color=colors['accent'], linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(ts, 0, wr, color=colors['accent'], alpha=0.15)
    ax.set_ylim(0, 1.05)
    if metrics['stage_boundaries']:
        add_stage_markers(ax, metrics['stage_boundaries'], 0, 1.05)
    
    # 2b. Mean Reward
    ax = axes[1]
    style_ax(ax, 'Mean Episode Reward', 'Timesteps', 'Reward')
    mr = metrics['eval_mean_rewards']
    ax.plot(ts, mr, color=colors['primary'], linewidth=2.5, marker='s', markersize=5)
    ax.axhline(y=0, color=colors['text'], linestyle='-', alpha=0.3)
    if metrics['stage_boundaries']:
        add_stage_markers(ax, metrics['stage_boundaries'], min(mr) if mr else 0, max(mr) if mr else 100)
    
    # 2c. Mean Length
    ax = axes[2]
    style_ax(ax, 'Mean Episode Length', 'Timesteps', 'Steps')
    ml = metrics['eval_mean_lengths']
    ax.plot(ts, ml, color=colors['secondary'], linewidth=2.5, marker='^', markersize=5)
    ax.axhline(y=40, color=colors['accent'], linestyle='--', alpha=0.5, label='Max steps')
    ax.legend(loc='upper right', facecolor=colors['bg'], edgecolor=colors['grid'], labelcolor=colors['text'])
    if metrics['stage_boundaries']:
        add_stage_markers(ax, metrics['stage_boundaries'], 0, max(ml) if ml else 40)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/evaluation_metrics.png', dpi=150, facecolor=colors['bg'])
    plt.close()
    print(f"  Saved: {save_dir}/evaluation_metrics.png")
    
    # =========================================================================
    # PLOT 3: Reward Distribution per Stage
    # =========================================================================
    if metrics['stage_boundaries']:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(colors['bg'])
        style_ax(ax, 'Reward Distribution by Difficulty Stage', 'Difficulty', 'Episode Reward')
        
        # Split rewards by stage
        boundaries = [0] + [b['timestep'] for b in metrics['stage_boundaries'][1:]] + [float('inf')]
        stage_rewards = [[] for _ in range(4)]
        
        cumulative_ts = 0
        episode_idx = 0
        ep_rewards = metrics['episode_rewards']
        ep_lengths = metrics['episode_lengths']
        
        # Approximate which episodes belong to which stage
        for i, (r, l) in enumerate(zip(ep_rewards, ep_lengths)):
            cumulative_ts += l
            stage = 0
            for j, boundary in enumerate(metrics['stage_boundaries']):
                if cumulative_ts >= boundary['timestep']:
                    stage = boundary['difficulty']
            if stage < 4:
                stage_rewards[stage].append(r)
        
        # Box plot
        positions = [0, 1, 2, 3]
        bp = ax.boxplot([sr if sr else [0] for sr in stage_rewards], 
                       positions=positions, widths=0.6, patch_artist=True)
        
        stage_names = ['Fixed Gold\n(Diff 0)', 'Random Gold\n(Diff 1)', 
                      '+ Wumpus\n(Diff 2)', '+ Pits\n(Diff 3)']
        ax.set_xticks(positions)
        ax.set_xticklabels(stage_names)
        
        for patch, color in zip(bp['boxes'], colors['stages']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set_color(colors['text'])
        for cap in bp['caps']:
            cap.set_color(colors['text'])
        for median in bp['medians']:
            median.set_color(colors['bg'])
            median.set_linewidth(2)
        for flier in bp['fliers']:
            flier.set_markerfacecolor(colors['text'])
            flier.set_alpha(0.5)
        
        ax.axhline(y=0, color=colors['text'], linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reward_distribution.png', dpi=150, facecolor=colors['bg'])
        plt.close()
        print(f"  Saved: {save_dir}/reward_distribution.png")
    
    # =========================================================================
    # PLOT 4: Final Summary Card
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    ax.axis('off')
    
    # Calculate final stats
    final_wr = metrics['eval_win_rates'][-1] if metrics['eval_win_rates'] else 0
    total_episodes = len(metrics['episode_rewards'])
    total_wins = sum(metrics['episode_wins'])
    avg_reward = np.mean(metrics['episode_rewards'][-100:]) if len(metrics['episode_rewards']) >= 100 else np.mean(metrics['episode_rewards'])
    avg_length = np.mean(metrics['episode_lengths'][-100:]) if len(metrics['episode_lengths']) >= 100 else np.mean(metrics['episode_lengths'])
    
    title = "Training Summary"
    ax.text(0.5, 0.92, title, ha='center', va='top', fontsize=20, 
           color=colors['accent'], fontweight='bold', transform=ax.transAxes)
    
    summary_text = f"""
    Total Episodes:        {total_episodes:,}
    Total Wins:            {total_wins:,}
    Overall Win Rate:      {total_wins/total_episodes*100:.1f}%
    
    Final Eval Win Rate:   {final_wr*100:.1f}%
    Avg Reward (last 100): {avg_reward:.1f}
    Avg Length (last 100): {avg_length:.1f} steps
    
    Curriculum Stages:     4
    Total Timesteps:       {metrics['eval_timesteps'][-1]:,}
    """
    
    ax.text(0.5, 0.45, summary_text, ha='center', va='center', fontsize=12,
           color=colors['text'], family='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#252545', edgecolor=colors['grid']))
    
    plt.savefig(f'{save_dir}/training_summary.png', dpi=150, facecolor=colors['bg'])
    plt.close()
    print(f"  Saved: {save_dir}/training_summary.png")
    
    return [f'{save_dir}/training_overview.png', 
            f'{save_dir}/evaluation_metrics.png',
            f'{save_dir}/reward_distribution.png',
            f'{save_dir}/training_summary.png']


def evaluate(model, difficulty, n=30):
    """Evaluate model win rate on given difficulty."""
    env = DummyVecEnv([lambda d=difficulty: WumpusWorldEnv(difficulty=d, max_steps=40)])
    wins = 0
    for _ in range(n):
        obs = env.reset()
        done = False
        for _ in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break
        if info[0].get('win'):
            wins += 1
    env.close()
    return wins / n


def train_curriculum(save_path='models/ppo_wumpus', plot=True):
    """
    Train with curriculum learning and generate training plots.
    
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
    
    # Initialize callback for metrics tracking
    callback = MetricsCallback(eval_freq=2000, eval_episodes=20)
    
    # Initialize on easiest level
    env = DummyVecEnv([lambda: WumpusWorldEnv(difficulty=0, max_steps=40)])
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
        (0, 10000, "Fixed gold (must explore)"),
        (1, 50000, "Random gold (must explore)"),
        (2, 80000, "+ Wumpus"),
        (3, 100000, "+ Pits (full game)"),
    ]
    
    for diff, steps, desc in stages:
        print(f"\nStage {diff + 1}: {desc} ({steps:,} steps)")
        
        callback.set_difficulty(diff)
        env = DummyVecEnv([lambda d=diff: WumpusWorldEnv(difficulty=d, max_steps=40)])
        model.set_env(env)
        model.learn(steps, callback=callback, progress_bar=True)
        
        win_rate = evaluate(model, diff)
        print(f"  Win rate: {win_rate * 100:.0f}%")
        env.close()
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    final_results = {}
    for d in range(4):
        rate = evaluate(model, d, n=50)
        labels = ["Fixed gold", "Random gold", "+ Wumpus", "Full game"]
        print(f"  {labels[d]}: {rate * 100:.0f}%")
        final_results[labels[d]] = rate
    
    model.save(save_path)
    print(f"\nModel saved: {save_path}")
    
    # Generate plots
    if plot:
        print("\nGenerating training plots...")
        metrics = callback.get_metrics()
        
        # Save metrics to JSON for later analysis
        os.makedirs('plots', exist_ok=True)
        with open('plots/metrics.json', 'w') as f:
            json.dump({k: [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                         for x in v] if isinstance(v, list) else v 
                      for k, v in metrics.items()}, f, indent=2, default=str)
        print(f"  Saved: plots/metrics.json")
        
        plot_training_metrics(metrics)
        print("\nAll plots saved to 'plots/' directory")
    
    return model


def quick_train(difficulty=0, timesteps=10000):
    """Quick training on single difficulty."""
    print(f"\nTraining on difficulty {difficulty} for {timesteps:,} steps")
    
    env = DummyVecEnv([lambda d=difficulty: WumpusWorldEnv(difficulty=d, max_steps=40)])
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
        env = WumpusWorldEnv(difficulty=difficulty, render_mode='ansi', max_steps=40)
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


def record_gif(model_path='models/ppo_wumpus', difficulty=1, save_path='recordings', n_episodes=10):
    """Record multiple episodes as GIFs."""
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    
    os.makedirs(save_path, exist_ok=True)
    model = PPO.load(model_path)
    
    wins = 0
    paths = []
    
    for ep in range(n_episodes):
        env = WumpusWorldEnv(difficulty=difficulty, render_mode='rgb_array', max_steps=40)
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
        result = "win" if info.get('win') else "loss"
        if info.get('win'):
            wins += 1
        gif_path = f'{save_path}/diff{difficulty}_ep{ep+1:02d}_{result}.gif'
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
        
        print(f"  [{ep+1:2d}/{n_episodes}] {gif_path} | {len(frames):2d} frames")
        paths.append(gif_path)
        env.close()
    
    print(f"\nDifficulty {difficulty}: {wins}/{n_episodes} wins ({wins*100//n_episodes}%)")
    return paths


def test_env():
    """Manual test of environment."""
    env = WumpusWorldEnv(difficulty=0, render_mode='ansi')
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print("Observation breakdown:")
    print(f"  [0-1]   Position:   row={obs[0]:.2f}, col={obs[1]:.2f}")
    print(f"  [2]     has_gold:   {obs[2]:.0f}")
    print(f"  [3]     can_win:    {obs[3]:.0f}")
    print(f"  [4-7]   Danger:     up={obs[4]:.0f} down={obs[5]:.0f} left={obs[6]:.0f} right={obs[7]:.0f}")
    print(f"  [8-11]  Glitter:    up={obs[8]:.0f} down={obs[9]:.0f} left={obs[10]:.0f} right={obs[11]:.0f}")
    print(f"  [12-15] Unvisited:  up={obs[12]:.0f} down={obs[13]:.0f} left={obs[14]:.0f} right={obs[15]:.0f}")
    print(f"\nActions: Up=0, Down=1, Left=2, Right=3, Climb=4")
    print(f"Gold is auto-picked when you step on it!")
    print(f"Glitter tells you WHICH DIRECTION the gold is!")
    env.render()
    
    # Difficulty 0: Gold is at [3,1], glitter_right=1, so Right (auto-pickup) -> Left -> Climb
    actions = [3, 2, 4]  # Right (pickup), Left, Climb
    
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


def plot_from_json(metrics_path='plots/metrics.json'):
    """Regenerate plots from saved metrics."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print("Regenerating plots from saved metrics...")
    plot_training_metrics(metrics)
    print("Done!")


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
            n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            record_gif(difficulty=diff, n_episodes=n_eps)
        elif cmd == "plot":
            # Regenerate plots from saved metrics
            plot_from_json()
        elif cmd.isdigit():
            quick_train(difficulty=int(cmd), timesteps=10000)
        else:
            print("Usage: python train.py [test|curriculum|watch|record|plot|0|1|2|3]")
            print("  curriculum      - Full training with plots")
            print("  record <d> [n]  - Record n episodes as GIFs")
            print("  plot            - Regenerate plots from saved metrics")
    else:
        train_curriculum()
