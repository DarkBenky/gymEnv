import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from getData import load_samples
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# ============================================================================
# CONFIGURATION - Edit hyperparameters here
# ============================================================================
CONFIG = {
    # Environment parameters
    "num_of_assets": 5,
    "window": 64,
    "initial_balance": 10000.0,
    "max_steps": 500,
    
    # Model architecture
    "policy_type": "MlpPolicy",
    "net_arch": [1024, 512, 256, 256, 128],  # Neural network layers [layer1, layer2, layer3]
    
    # PPO hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # Training parameters
    "total_timesteps": 200000,
    "n_eval_episodes": 10,
}
# ============================================================================


class WandbCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_returns = []
        self.episode_values = []
        
    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            wandb.log({
                "train/episode_reward": self.model.ep_info_buffer[-1]['r'],
                "train/episode_length": self.model.ep_info_buffer[-1]['l'],
                "train/timestep": self.num_timesteps,
            })
        
        # Log custom info from environment if available
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'total_value' in info:
                self.episode_returns.append(info['return'])
                self.episode_values.append(info['total_value'])
                
                wandb.log({
                    "env/total_value": info['total_value'],
                    "env/balance": info['balance'],
                    "env/pnl": info['pnl'],
                    "env/return_pct": info['return'] * 100,
                    "train/timestep": self.num_timesteps,
                })
        
        return True


class CustomEnv(gym.Env):
    """Crypto Trading Environment that follows gym interface."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_of_assets=10, window=64, initial_balance=10000.0, render_mode=None, mode='train'):
        super().__init__()
        
        self.num_of_assets = num_of_assets
        self.window = window
        self.initial_balance = initial_balance
        self.mode = mode
        
        # Load all samples once during initialization
        print(f"Loading market data ({mode} mode)...")
        self.X_data, self.Y_data = load_samples(batches=1024, num_of_assets=num_of_assets, window=window, mode=mode)
        # X_data shape: (batches, num_assets, window, num_features)
        # Y_data shape: (batches, num_assets) - next price change
        
        self.num_batches = self.X_data.shape[0]
        self.actual_num_assets = self.X_data.shape[1]
        self.num_features = self.X_data.shape[3]
        
        print(f"Loaded {self.num_batches} samples with {self.actual_num_assets} assets")
        
        # Action space: for each asset, predict position [-1 to 1]
        # -1 = full short, 0 = hold/no position, 1 = full long
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.actual_num_assets,), dtype=np.float32
        )
        
        # Observation space: flattened window data + portfolio state
        # Window data: (num_assets * window * num_features)
        # Portfolio state: balance + positions (num_assets) + total_value
        obs_size = self.actual_num_assets * self.window * self.num_features + self.actual_num_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 200
        self.current_batch_idx = 0
        
        # Portfolio state
        self.balance = initial_balance
        self.positions = np.zeros(self.actual_num_assets, dtype=np.float32)  # Amount held in each asset
        self.total_value = initial_balance
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.positions = np.zeros(self.actual_num_assets, dtype=np.float32)
        self.total_value = self.initial_balance
        self.current_step = 0
        
        # Select random batch
        self.current_batch_idx = np.random.randint(0, self.num_batches)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Get current price changes (targets for this step)
        price_changes = self.Y_data[self.current_batch_idx]  # Shape: (num_assets,)
        
        # Calculate portfolio value before action
        old_value = self.total_value
        
        # Execute trades based on action
        # action[i] represents desired position in asset i as fraction of total portfolio
        desired_positions = action * self.total_value
        
        # Calculate position changes (simple execution, no fees for now)
        position_changes = desired_positions - self.positions
        
        # Update balance and positions (cash required/freed)
        self.balance -= np.sum(position_changes)
        self.positions = desired_positions.copy()
        
        # Move to next time step
        self.current_step += 1
        
        # Apply price changes to positions
        # positions * price_change gives profit/loss for each asset
        pnl = self.positions * price_changes
        total_pnl = np.sum(pnl)
        
        # Update positions value after price change
        self.positions = self.positions * (1 + price_changes)
        
        # Calculate new total value
        self.total_value = self.balance + np.sum(self.positions)
        
        # Reward: profit/loss as percentage of initial portfolio
        reward = (self.total_value - old_value) / self.initial_balance
        
        # Penalize if balance goes negative (over-leveraged)
        if self.balance < 0:
            reward -= 0.1
        
        # Check if episode is done
        terminated = bool(self.total_value <= 0)  # Bankruptcy
        truncated = self.current_step >= self.max_steps
        
        # Get next observation (move to next batch for next step)
        if not terminated and not truncated:
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
        
        obs = self._get_observation()
        
        # Additional info
        info = {
            "steps": self.current_step,
            "total_value": self.total_value,
            "balance": self.balance,
            "pnl": total_pnl,
            "return": (self.total_value - self.initial_balance) / self.initial_balance
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construct observation from current state."""
        # Get current market window
        window_data = self.X_data[self.current_batch_idx]  # Shape: (num_assets, window, num_features)
        
        # Flatten window data
        flattened_window = window_data.flatten()
        
        # Portfolio state: [balance, positions..., total_value]
        portfolio_state = np.concatenate([
            [self.balance / self.initial_balance],  # Normalized balance
            self.positions / self.initial_balance,   # Normalized positions
            [self.total_value / self.initial_balance]  # Normalized total value
        ])
        
        # Combine all into single observation
        obs = np.concatenate([flattened_window, portfolio_state]).astype(np.float32)
        
        return obs
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
                  f"Total Value: ${self.total_value:.2f}, "
                  f"Return: {((self.total_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
    
    def close(self):
        """Clean up resources."""
        pass


# Create and check the environment
if __name__ == "__main__":
    # Initialize Weights & Biases
    wandb.init(
        project="crypto-trading-rl",
        config=CONFIG,
        name="ppo-crypto-trading",
    )
    
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)
    
    # Create TRAINING environment
    env = CustomEnv(
        num_of_assets=CONFIG["num_of_assets"], 
        window=CONFIG["window"], 
        initial_balance=CONFIG["initial_balance"],
        mode='train'
    )
    env.max_steps = CONFIG["max_steps"]
    
    # Check if environment follows Gym API
    print("\nChecking environment...")
    check_env(env, warn=True)
    
    # Train PPO model
    print("\nTraining PPO model...")
    print(f"Network architecture: {CONFIG['net_arch']}")
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    
    model = PPO(
        CONFIG["policy_type"],
        env,
        verbose=1,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        clip_range_vf=CONFIG["clip_range_vf"],
        ent_coef=CONFIG["ent_coef"],
        vf_coef=CONFIG["vf_coef"],
        max_grad_norm=CONFIG["max_grad_norm"],
        policy_kwargs=dict(net_arch=CONFIG["net_arch"]),
    )
    
    # Create wandb callback
    wandb_callback = WandbCallback()
    
    # Train the agent
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=wandb_callback)
    
    # Save the model
    model.save("ppo_crypto_trading")
    print("\nModel saved as 'ppo_crypto_trading'")
    
    # Log model artifact to wandb
    artifact = wandb.Artifact('ppo_crypto_trading', type='model')
    artifact.add_file('ppo_crypto_trading.zip')
    wandb.log_artifact(artifact)
    
    # Evaluate the trained model on TEST data (no data leakage!)
    print("\nEvaluating trained model on TEST set...")
    env.close()
    
    # Create TEST environment with unseen data
    test_env = CustomEnv(
        num_of_assets=CONFIG["num_of_assets"], 
        window=CONFIG["window"], 
        initial_balance=CONFIG["initial_balance"],
        mode='test'
    )
    test_env.max_steps = CONFIG["max_steps"]
    
    eval_episodes = CONFIG["n_eval_episodes"]
    all_returns = []
    all_values = []
    all_balances = []
    episode_steps = []
    
    for episode in range(eval_episodes):
        obs, info = test_env.reset()
        episode_return = []
        episode_value = []
        episode_balance = []
        done = False
        step = 0
        
        while not done and step < CONFIG["max_steps"]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            episode_return.append(info['return'] * 100)
            episode_value.append(info['total_value'])
            episode_balance.append(info['balance'])
            
            done = terminated or truncated
            step += 1
        
        all_returns.append(episode_return)
        all_values.append(episode_value)
        all_balances.append(episode_balance)
        episode_steps.append(step)
        
        print(f"Episode {episode + 1}/{eval_episodes}: "
              f"Final Value=${info['total_value']:.2f}, "
              f"Return={info['return']*100:.2f}%, "
              f"Steps={step}")
    
    # Create performance charts
    print("\nCreating performance charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trading Agent Performance', fontsize=16, fontweight='bold')
    
    # Plot all episodes' portfolio values
    for i, values in enumerate(all_values):
        axes[0, 0].plot(values, alpha=0.6, label=f'Episode {i+1}')
    axes[0, 0].axhline(y=CONFIG["initial_balance"], color='r', linestyle='--', label='Initial Balance')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best', fontsize=8)
    
    # Chart 2: Returns Over Time
    for i, returns in enumerate(all_returns):
        axes[0, 1].plot(returns, alpha=0.6, label=f'Episode {i+1}')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Break-even')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].set_title('Returns Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='best', fontsize=8)
    
    # Chart 3: Final Returns Distribution
    final_returns = [returns[-1] for returns in all_returns]
    axes[1, 0].bar(range(1, eval_episodes + 1), final_returns, 
                    color=['g' if r > 0 else 'r' for r in final_returns])
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Final Return (%)')
    axes[1, 0].set_title('Final Returns by Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Chart 4: Statistics Summary
    axes[1, 1].axis('off')
    stats_text = f"""
    Performance Statistics
    {'='*30}
    
    Episodes: {eval_episodes}
    
    Returns:
      Mean: {np.mean(final_returns):.2f}%
      Median: {np.median(final_returns):.2f}%
      Std Dev: {np.std(final_returns):.2f}%
      Min: {np.min(final_returns):.2f}%
      Max: {np.max(final_returns):.2f}%
    
    Final Values:
      Mean: ${np.mean([v[-1] for v in all_values]):.2f}
      Min: ${np.min([v[-1] for v in all_values]):.2f}
      Max: ${np.max([v[-1] for v in all_values]):.2f}
    
    Win Rate: {sum(1 for r in final_returns if r > 0)/eval_episodes*100:.1f}%
    
    Avg Steps: {np.mean(episode_steps):.1f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight')
    print("Performance chart saved as 'performance_chart.png'")
    
    # Log chart to wandb
    wandb.log({"performance_chart": wandb.Image('performance_chart.png')})
    
    # Log summary statistics
    wandb.log({
        "eval/mean_return": np.mean(final_returns),
        "eval/median_return": np.median(final_returns),
        "eval/std_return": np.std(final_returns),
        "eval/min_return": np.min(final_returns),
        "eval/max_return": np.max(final_returns),
        "eval/mean_final_value": np.mean([v[-1] for v in all_values]),
        "eval/win_rate": sum(1 for r in final_returns if r > 0)/eval_episodes*100,
        "eval/avg_steps": np.mean(episode_steps),
    })
    
    print(f"\nMean Return: {np.mean(final_returns):.2f}%")
    print(f"Win Rate: {sum(1 for r in final_returns if r > 0)/eval_episodes*100:.1f}%")
    
    env.close()
    test_env.close()
    wandb.finish()
    
    print("\n✅ Training complete! Check W&B dashboard for detailed metrics.")
    print(f"   https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
