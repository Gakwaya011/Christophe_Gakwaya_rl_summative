import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import SimpleAthleteNutritionEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class TrainingCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # Save progress every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                self.model.save(f"{self.save_path}/model_{len(self.episode_rewards)}")
                
        return True

def train_dqn_hyperparams():
    """Train DQN with 10 different hyperparameter combinations"""
    
    results = []
    
    # 10 Hyperparameter combinations for DQN
    hyperparams = [
        {'learning_rate': 1e-3, 'buffer_size': 10000, 'batch_size': 32, 'gamma': 0.99, 'exploration_fraction': 0.3},
        {'learning_rate': 5e-4, 'buffer_size': 50000, 'batch_size': 64, 'gamma': 0.95, 'exploration_fraction': 0.2},
        {'learning_rate': 1e-4, 'buffer_size': 100000, 'batch_size': 128, 'gamma': 0.9, 'exploration_fraction': 0.4},
        {'learning_rate': 2e-3, 'buffer_size': 20000, 'batch_size': 16, 'gamma': 0.99, 'exploration_fraction': 0.25},
        {'learning_rate': 1e-3, 'buffer_size': 50000, 'batch_size': 32, 'gamma': 0.97, 'exploration_fraction': 0.35},
        {'learning_rate': 7e-4, 'buffer_size': 75000, 'batch_size': 64, 'gamma': 0.94, 'exploration_fraction': 0.15},
        {'learning_rate': 3e-4, 'buffer_size': 100000, 'batch_size': 128, 'gamma': 0.92, 'exploration_fraction': 0.5},
        {'learning_rate': 8e-4, 'buffer_size': 25000, 'batch_size': 48, 'gamma': 0.96, 'exploration_fraction': 0.3},
        {'learning_rate': 4e-4, 'buffer_size': 80000, 'batch_size': 96, 'gamma': 0.93, 'exploration_fraction': 0.4},
        {'learning_rate': 6e-4, 'buffer_size': 60000, 'batch_size': 80, 'gamma': 0.98, 'exploration_fraction': 0.2}
    ]
    
    for run_id, params in enumerate(hyperparams):
        print(f"\n=== DQN Run {run_id + 1}/10 ===")
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
        
        # Train model
        callback = TrainingCallback(save_dir)
        model.learn(
            total_timesteps=50000,
            callback=callback,
            log_interval=1000
        )
        
        # Save final model
        model.save(f"{save_dir}/final_model")
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        # Store results
        result = {
            'run_id': run_id,
            'algorithm': 'DQN',
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'learning_rate': params['learning_rate'],
            'buffer_size': params['buffer_size'],
            'batch_size': params['batch_size'],
            'gamma': params['gamma'],
            'exploration_fraction': params['exploration_fraction'],
            'episode_rewards': callback.episode_rewards
        }
        results.append(result)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        env.close()
    
    # Save results to CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episode_rewards'} for r in results])
    df.to_csv('./results/dqn_results.csv', index=False)
    
    # Save training curves
    training_curves = []
    for i, result in enumerate(results):
        for episode, reward in enumerate(result['episode_rewards']):
            training_curves.append({
                'run_id': i,
                'algorithm': 'DQN',
                'episode': episode,
                'reward': reward
            })
    
    training_df = pd.DataFrame(training_curves)
    training_df.to_csv('./results/dqn_training_curves.csv', index=False)
    
    return results

if __name__ == "__main__":
    os.makedirs('./models/dqn', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    train_dqn_hyperparams()