import os
import sys
import numpy as np
import pandas as pd

# FIX: Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment.custom_env import SimpleAthleteNutritionEnv

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.saved_log_probs = []
        self.rewards = []

def train_reinforce_hyperparams():
    """Train REINFORCE with 10 different hyperparameter combinations"""
    results = []
    
    hyperparams = [
        {'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_size': 64},
        {'learning_rate': 5e-4, 'gamma': 0.95, 'hidden_size': 128},
        {'learning_rate': 2e-3, 'gamma': 0.98, 'hidden_size': 32},
        {'learning_rate': 7e-4, 'gamma': 0.97, 'hidden_size': 256},
        {'learning_rate': 3e-4, 'gamma': 0.96, 'hidden_size': 64},
        {'learning_rate': 1e-2, 'gamma': 0.99, 'hidden_size': 512},
        {'learning_rate': 8e-4, 'gamma': 0.94, 'hidden_size': 128},
        {'learning_rate': 4e-4, 'gamma': 0.93, 'hidden_size': 256},
        {'learning_rate': 6e-4, 'gamma': 0.95, 'hidden_size': 64},
        {'learning_rate': 9e-4, 'gamma': 0.98, 'hidden_size': 128}
    ]
    
    for run_id, params in enumerate(hyperparams):
        print(f"\n=== REINFORCE Run {run_id + 1}/10 ===")
        
        env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = REINFORCE(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=params['learning_rate'],
            gamma=params['gamma']
        )
        agent.policy.fc1 = nn.Linear(state_dim, params['hidden_size'])
        agent.policy.fc2 = nn.Linear(params['hidden_size'], params['hidden_size'])
        agent.policy.fc3 = nn.Linear(params['hidden_size'], action_dim)
        
        # Training
        episode_rewards = []
        for episode in range(500):  # 500 episodes
            state, _ = env.reset()
            episode_reward = 0
            
            for t in range(100):  # Max steps per episode
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.rewards.append(reward)
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            agent.update_policy()
            episode_rewards.append(episode_reward)
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Evaluation
        eval_rewards = []
        for _ in range(10):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        result = {
            'run_id': run_id,
            'algorithm': 'REINFORCE',
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            **params,
            'episode_rewards': episode_rewards
        }
        results.append(result)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        env.close()
    
    # Save results
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episode_rewards'} for r in results])
    df.to_csv('./results/reinforce_results.csv', index=False)
    
    training_curves = []
    for i, result in enumerate(results):
        for episode, reward in enumerate(result['episode_rewards']):
            training_curves.append({
                'run_id': i, 'algorithm': 'REINFORCE', 'episode': episode, 'reward': reward
            })
    
    pd.DataFrame(training_curves).to_csv('./results/reinforce_training_curves.csv', index=False)
    return results

if __name__ == "__main__":
    os.makedirs('./results', exist_ok=True)
    train_reinforce_hyperparams()