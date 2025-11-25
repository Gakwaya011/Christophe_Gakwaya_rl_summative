import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import SimpleAthleteNutritionEnv

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
            
        return True

def train_ppo_hyperparams():
    """Train PPO with 10 different hyperparameter combinations"""
    results = []
    
    hyperparams = [
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99},
        {'learning_rate': 1e-3, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 5, 'gamma': 0.95},
        {'learning_rate': 5e-4, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 15, 'gamma': 0.98},
        {'learning_rate': 2e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 20, 'gamma': 0.97},
        {'learning_rate': 7e-4, 'n_steps': 512, 'batch_size': 16, 'n_epochs': 8, 'gamma': 0.96},
        {'learning_rate': 4e-4, 'n_steps': 3072, 'batch_size': 96, 'n_epochs': 12, 'gamma': 0.99},
        {'learning_rate': 6e-4, 'n_steps': 1536, 'batch_size': 48, 'n_epochs': 6, 'gamma': 0.94},
        {'learning_rate': 8e-4, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 4, 'gamma': 0.93},
        {'learning_rate': 3e-4, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 25, 'gamma': 0.98},
        {'learning_rate': 9e-4, 'n_steps': 768, 'batch_size': 24, 'n_epochs': 3, 'gamma': 0.95}
    ]
    
    for run_id, params in enumerate(hyperparams):
        print(f"\n=== PPO Run {run_id + 1}/10 ===")
        
        env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
        env = Monitor(env)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            verbose=0,
            tensorboard_log="./logs/ppo/"
        )
        
        save_dir = f"./models/ppo/run_{run_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        callback = TrainingCallback(save_dir)
        model.learn(total_timesteps=50000, callback=callback)
        model.save(f"{save_dir}/final_model")
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        result = {
            'run_id': run_id,
            'algorithm': 'PPO',
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            **params,
            'episode_rewards': callback.episode_rewards
        }
        results.append(result)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        env.close()
    
    # Save results
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episode_rewards'} for r in results])
    df.to_csv('./results/ppo_results.csv', index=False)
    
    training_curves = []
    for i, result in enumerate(results):
        for episode, reward in enumerate(result['episode_rewards']):
            training_curves.append({
                'run_id': i, 'algorithm': 'PPO', 'episode': episode, 'reward': reward
            })
    
    pd.DataFrame(training_curves).to_csv('./results/ppo_training_curves.csv', index=False)
    return results

def train_a2c_hyperparams():
    """Train A2C with 10 different hyperparameter combinations"""
    results = []
    
    hyperparams = [
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 0.95},
        {'learning_rate': 3e-4, 'n_steps': 10, 'gamma': 0.95, 'gae_lambda': 0.90},
        {'learning_rate': 1e-3, 'n_steps': 8, 'gamma': 0.98, 'gae_lambda': 0.92},
        {'learning_rate': 5e-4, 'n_steps': 12, 'gamma': 0.97, 'gae_lambda': 0.88},
        {'learning_rate': 2e-3, 'n_steps': 4, 'gamma': 0.96, 'gae_lambda': 0.85},
        {'learning_rate': 8e-4, 'n_steps': 6, 'gamma': 0.94, 'gae_lambda': 0.91},
        {'learning_rate': 4e-4, 'n_steps': 15, 'gamma': 0.99, 'gae_lambda': 0.93},
        {'learning_rate': 6e-4, 'n_steps': 7, 'gamma': 0.93, 'gae_lambda': 0.87},
        {'learning_rate': 9e-4, 'n_steps': 9, 'gamma': 0.95, 'gae_lambda': 0.89},
        {'learning_rate': 1e-4, 'n_steps': 20, 'gamma': 0.98, 'gae_lambda': 0.94}
    ]
    
    for run_id, params in enumerate(hyperparams):
        print(f"\n=== A2C Run {run_id + 1}/10 ===")
        
        env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
        env = Monitor(env)
        
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            verbose=0,
            tensorboard_log="./logs/a2c/"
        )
        
        save_dir = f"./models/a2c/run_{run_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        callback = TrainingCallback(save_dir)
        model.learn(total_timesteps=50000, callback=callback)
        model.save(f"{save_dir}/final_model")
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        result = {
            'run_id': run_id,
            'algorithm': 'A2C',
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            **params,
            'episode_rewards': callback.episode_rewards
        }
        results.append(result)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        env.close()
    
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episode_rewards'} for r in results])
    df.to_csv('./results/a2c_results.csv', index=False)
    
    training_curves = []
    for i, result in enumerate(results):
        for episode, reward in enumerate(result['episode_rewards']):
            training_curves.append({
                'run_id': i, 'algorithm': 'A2C', 'episode': episode, 'reward': reward
            })
    
    pd.DataFrame(training_curves).to_csv('./results/a2c_training_curves.csv', index=False)
    return results

if __name__ == "__main__":
    os.makedirs('./models/ppo', exist_ok=True)
    os.makedirs('./models/a2c', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    print("Training PPO...")
    ppo_results = train_ppo_hyperparams()
    
    print("\nTraining A2C...")
    a2c_results = train_a2c_hyperparams()