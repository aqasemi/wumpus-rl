import time
import sys
from wumpus_env import WumpusWorldEnv


def play_interactive():
    """Play the game interactively using keyboard."""
    print("\n" + "=" * 50)
    print("  WUMPUS WORLD - Interactive Mode")
    print("=" * 50)
    print("\nControls:")
    print("  W / ↑  : Move Up")
    print("  S / ↓  : Move Down")
    print("  A / ←  : Move Left")
    print("  D / →  : Move Right")
    print("  Q      : Quit")
    print("\nGoal: Find the gold (G) and avoid pits (P) and Wumpus (W)!")
    print("=" * 50 + "\n")
    
    env = WumpusWorldEnv(render_mode='human')
    obs, _ = env.reset()
    
    total_reward = 0
    done = False
    
    action_map = {
        'w': 0, 'up': 0,      # Up
        's': 1, 'down': 1,    # Down
        'a': 2, 'left': 2,    # Left
        'd': 3, 'right': 3    # Right
    }
    
    import pygame
    
    while not done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                action = None
                
                if event.key in [pygame.K_w, pygame.K_UP]:
                    action = 0
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    action = 1
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    action = 2
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    action = 3
                elif event.key == pygame.K_q:
                    env.close()
                    return
                
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    
                    if done:
                        if info.get('win'):
                            print("\n🎉 VICTORY! You found the gold!")
                        else:
                            print("\n💀 GAME OVER!")
                        print(f"Total Reward: {total_reward}")
        
        time.sleep(0.05)  # Small delay to prevent CPU spinning
    
    # Wait for user to close
    print("\nPress Q or close window to exit...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                waiting = False
        time.sleep(0.05)
    
    env.close()


def play_random():
    """Watch a random agent play."""
    print("\nWatching random agent play...")
    
    env = WumpusWorldEnv(render_mode='human')
    obs, _ = env.reset()
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1
        time.sleep(0.3)
    
    print(f"\nGame ended after {step} steps")
    print(f"Total Reward: {total_reward}")
    print(f"Result: {'WIN!' if info.get('win') else 'Lost'}")
    
    time.sleep(2)
    env.close()


def test_env():
    """Quick test of the environment."""
    print("\nTesting environment...")
    
    env = WumpusWorldEnv(render_mode='ansi')
    
    print("\nInitial state:")
    obs, _ = env.reset()
    env.render()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take a few random steps
    print("\nTaking 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        action_name = env.action_names[action]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Action = {action_name}, Reward = {reward}")
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended: {'Win!' if info.get('win') else 'Lost'}")
            break
    
    env.close()
    print("\nEnvironment test complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            play_interactive()
        elif sys.argv[1] == "random":
            play_random()
        elif sys.argv[1] == "test":
            test_env()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python main.py [play|random|test]")
    else:
        print("Wumpus World")
        print("=" * 40)
        print("\nCommands:")
        print("  python main.py play   - Play interactively")
        print("  python main.py random - Watch random agent")
        print("  python main.py test   - Test environment")
        print("\nFor training:")
        print("  python train.py       - Train PPO agent")
        print("  python train.py play  - Watch trained agent")
        print("  python train.py demo  - Random agent demo")
