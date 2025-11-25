# training/dqn_training_fixed.py
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environment.custom_env import SimpleAthleteNutritionEnv

class SaveTrainingProgress(BaseCallback):
    """
    Custom callback for saving training progress
    """
    def __init__(self, save_path, verbose=0):
        super(SaveTrainingProgress, self).__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get reward and done from info
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            self.current_episode_reward = 0
            
            # Save progress every 10 episodes
            if self.episode_count % 10 == 0:
                # Save current rewards
                rewards_df = pd.DataFrame({
                    'episode': range(len(self.episode_rewards)),
                    'reward': self.episode_rewards
                })
                rewards_df.to_csv(f"{self.save_path}/training_progress.csv", index=False)
                
        return True

def train_dqn_with_progress():
    """Train DQN with proper progress tracking"""
    
    results = []
    all_training_curves = []
    
    hyperparams = [
        {'learning_rate': 1e-3, 'buffer_size': 10000, 'batch_size': 32, 'gamma': 0.99, 'exploration_fraction': 0.3},
        {'learning_rate': 5e-4, 'buffer_size': 50000, 'batch_size': 64, 'gamma': 0.95, 'exploration_fraction': 0.2},
        {'learning_rate': 1e-4, 'buffer_size': 100000, 'batch_size': 128, 'gamma': 0.9, 'exploration_fraction': 0.4},
        {'learning_rate': 2e-3, 'buffer_size': 20000, 'batch_size': 16, 'gamma': 0.99, 'exploration_fraction': 0.25},
        {'learning_rate': 1e-3, 'buffer_size': 50000, 'batch_size': 32, 'gamma': 0.97, 'exploration_fraction': 0.35},
    ]
    
    for run_id, params in enumerate(hyperparams):
        print(f"\n=== DQN Run {run_id + 1}/{len(hyperparams)} ===")
        print(f"Hyperparameters: {params}")
        
        # Create environment
        env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
        env = Monitor(env)
        
        # Create model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            exploration_fraction=params['exploration_fraction'],
            verbose=0,
            tensorboard_log="./logs/dqn/"
        )
        
        # Create save directory
        save_dir = f"./models/dqn/run_{run_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Train model with progress tracking
        callback = SaveTrainingProgress(save_dir)
        model.learn(
            total_timesteps=20000,  # Reduced for faster testing
            callback=callback,
            log_interval=1000
        )
        
        # Save final model
        model.save(f"{save_dir}/final_model")
        
        # Save training curves
        if callback.episode_rewards:
            for episode, reward in enumerate(callback.episode_rewards):
                all_training_curves.append({
                    'run_id': run_id,
                    'algorithm': 'dqn',
                    'episode': episode,
                    'reward': reward
                })
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        
        # Store results
        result = {
            'run_id': run_id,
            'algorithm': 'DQN',
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            **params
        }
        results.append(result)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Episodes trained: {len(callback.episode_rewards)}")
        
        env.close()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('./results/dqn_results.csv', index=False)
    
    # Save training curves
    if all_training_curves:
        curves_df = pd.DataFrame(all_training_curves)
        curves_df.to_csv('./results/dqn_training_curves.csv', index=False)
        print(f"Saved {len(curves_df)} training data points")
    
    return results

if __name__ == "__main__":
    os.makedirs('./models/dqn', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    train_dqn_with_progress()