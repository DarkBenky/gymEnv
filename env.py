import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from getData import load_samples
import wandb


class WandbCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
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
                wandb.log({
                    "env/total_value": info['total_value'],
                    "env/balance": info['balance'],
                    "env/pnl": info['pnl'],
                    "env/return": info['return'] * 100,
                    "train/timestep": self.num_timesteps,
                })
        
        return True


class CustomEnv(gym.Env):
    """Crypto Trading Environment that follows gym interface."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_of_assets=10, window=64, initial_balance=10000.0, render_mode=None):
        super().__init__()
        
        self.num_of_assets = num_of_assets
        self.window = window
        self.initial_balance = initial_balance
        
        # Load all samples once during initialization
        print("Loading market data...")
        self.X_data, self.Y_data = load_samples(batches=1024, num_of_assets=num_of_assets, window=window)
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
        config={
            "num_of_assets": 5,
            "window": 64,
            "initial_balance": 10000.0,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "total_timesteps": 50000,
        },
        name="ppo-crypto-trading",
    )
    
    # Create environment with 5 assets
    env = CustomEnv(num_of_assets=5, window=64, initial_balance=10000.0)
    
    # Check if environment follows Gym API
    print("Checking environment...")
    check_env(env, warn=True)
    
    # Train PPO model
    print("\nTraining PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=wandb.config.learning_rate,
        n_steps=wandb.config.n_steps,
        batch_size=wandb.config.batch_size,
        n_epochs=wandb.config.n_epochs,
        gamma=wandb.config.gamma,
        gae_lambda=wandb.config.gae_lambda,
        clip_range=wandb.config.clip_range,
        ent_coef=wandb.config.ent_coef,
    )
    
    # Create wandb callback
    wandb_callback = WandbCallback()
    
    # Train the agent
    model.learn(total_timesteps=wandb.config.total_timesteps, callback=wandb_callback)
    
    # Save the model
    model.save("ppo_crypto_trading")
    print("\nModel saved as 'ppo_crypto_trading'")
    
    # Log model artifact to wandb
    artifact = wandb.Artifact('ppo_crypto_trading', type='model')
    artifact.add_file('ppo_crypto_trading.zip')
    wandb.log_artifact(artifact)
    
    # Test the trained model
    print("\nTesting trained model...")
    obs, info = env.reset()
    total_return = 0
    test_metrics = {
        'final_value': 0,
        'final_return': 0,
        'steps_completed': 0
    }
    
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:  # Print every 10 steps
            print(f"Step {i}: Value=${info['total_value']:.2f}, Return={info['return']*100:.2f}%, PnL=${info['pnl']:.2f}")
            wandb.log({
                "test/value": info['total_value'],
                "test/return": info['return'] * 100,
                "test/step": i,
            })
        
        if terminated or truncated:
            print(f"\nEpisode finished after {i+1} steps")
            print(f"Final Value: ${info['total_value']:.2f}")
            print(f"Total Return: {info['return']*100:.2f}%")
            
            test_metrics['final_value'] = info['total_value']
            test_metrics['final_return'] = info['return'] * 100
            test_metrics['steps_completed'] = i + 1
            break
    
    # Log final test metrics
    wandb.log({
        "test/final_value": test_metrics['final_value'],
        "test/final_return": test_metrics['final_return'],
        "test/steps_completed": test_metrics['steps_completed'],
    })
    
    env.close()
    wandb.finish()
    
    # Load and use the model later
    # model = PPO.load("ppo_crypto_trading")
    # Load and use the model later
    # model = PPO.load("ppo_custom_env")