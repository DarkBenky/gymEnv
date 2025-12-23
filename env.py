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
matplotlib.use('Agg')


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "num_of_assets": 5,
    "window": 64,
    "initial_balance": 10000.0,
    "max_steps": 500,
    "val_asset_ratio": 0.2,
    "data_seed": 42,
    
    "policy_type": "MlpPolicy",
    "net_arch": [4096, 2048, 1024, 512, 256, 256, 128],
    
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
    
    "total_timesteps": 200000,
    "n_eval_episodes": 10,
}
# ============================================================================


class WandbCallback(BaseCallback):
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_of_assets=10, window=64, initial_balance=10000.0, render_mode=None, mode='train', val_asset_ratio=0.2, data_seed=42):
        super().__init__()
        
        self.num_of_assets = num_of_assets
        self.window = window
        self.initial_balance = initial_balance
        self.mode = mode
        
        print(f"Loading market data ({mode} mode)...")
        self.X_data, self.Y_data = load_samples(
            batches=1024, 
            num_of_assets=num_of_assets, 
            window=window, 
            mode=mode,
            val_asset_ratio=val_asset_ratio,
            seed=data_seed
        )
        
        self.num_batches = self.X_data.shape[0]
        self.actual_num_assets = self.X_data.shape[1]
        self.actual_num_features = self.X_data.shape[3]
        
        if self.actual_num_assets != num_of_assets:
            raise ValueError(f"Asset count mismatch: requested {num_of_assets} assets but loaded {self.actual_num_assets}. Check that all data files exist and are loadable.")
        
        print(f"Loaded {self.num_batches} samples with {self.actual_num_assets} assets and {self.actual_num_features} features")
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.actual_num_assets,), dtype=np.float32
        )
        
        obs_size = self.actual_num_assets * self.window * self.actual_num_features + self.actual_num_assets + 2
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
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.positions = np.zeros(self.actual_num_assets, dtype=np.float32)
        self.total_value = self.initial_balance
        self.current_step = 0
        
        self.current_batch_idx = np.random.randint(0, self.num_batches)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        price_changes = self.Y_data[self.current_batch_idx]
        
        old_value = self.total_value
        
        desired_positions = action * self.total_value
        position_changes = desired_positions - self.positions
        
        self.balance -= np.sum(position_changes)
        self.positions = desired_positions.copy()
        
        self.current_step += 1
        
        pnl = self.positions * price_changes
        total_pnl = np.sum(pnl)
        
        self.positions = self.positions * (1 + price_changes)
        self.total_value = self.balance + np.sum(self.positions)
        
        reward = (self.total_value - old_value) / self.initial_balance
        
        if self.balance < 0:
            reward -= 0.1
        
        terminated = bool(self.total_value <= 0)
        truncated = self.current_step >= self.max_steps
        
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
        window_data = self.X_data[self.current_batch_idx]
        flattened_window = window_data.flatten()
        
        portfolio_state = np.concatenate([
            [self.balance / self.initial_balance],
            self.positions / self.initial_balance,
            [self.total_value / self.initial_balance]
        ])
        
        obs = np.concatenate([flattened_window, portfolio_state]).astype(np.float32)
        return obs
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
                  f"Total Value: ${self.total_value:.2f}, "
                  f"Return: {((self.total_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
    
    def close(self):
        pass


if __name__ == "__main__":
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
    
    env = CustomEnv(
        num_of_assets=CONFIG["num_of_assets"], 
        window=CONFIG["window"], 
        initial_balance=CONFIG["initial_balance"],
        mode='train',
        val_asset_ratio=CONFIG["val_asset_ratio"],
        data_seed=CONFIG["data_seed"]
    )
    env.max_steps = CONFIG["max_steps"]
    
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
    
    print("\\nEvaluating trained model on VALIDATION set (unseen assets)...")
    env.close()
    
    test_env = CustomEnv(
        num_of_assets=CONFIG["num_of_assets"], 
        window=CONFIG["window"], 
        initial_balance=CONFIG["initial_balance"],
        mode='val',
        val_asset_ratio=CONFIG["val_asset_ratio"],
        data_seed=CONFIG["data_seed"]
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
    print("\\nCreating performance charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trading Agent Performance (Validation on Unseen Assets)', fontsize=16, fontweight='bold')
    
    # Plot all episodes' portfolio values
    for i, values in enumerate(all_values):
        axes[0, 0].plot(values, alpha=0.6, label=f'Episode {i+1}')
    axes[0, 0].axhline(y=CONFIG["initial_balance"], color='r', linestyle='--', label='Initial Balance')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best', fontsize=8)
    
    for i, returns in enumerate(all_returns):
        axes[0, 1].plot(returns, alpha=0.6, label=f'Episode {i+1}')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Break-even')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].set_title('Returns Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='best', fontsize=8)
    
    final_returns = [returns[-1] for returns in all_returns]
    axes[1, 0].bar(range(1, eval_episodes + 1), final_returns, 
                    color=['g' if r > 0 else 'r' for r in final_returns])
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Final Return (%)')
    axes[1, 0].set_title('Final Returns by Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
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
    
    wandb.log({"performance_chart": wandb.Image('performance_chart.png')})
    
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
