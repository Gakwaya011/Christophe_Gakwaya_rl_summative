# quick_test.py
import sys
import os
sys.path.append('/home/chris/Videos/Christophe_Gakwaya_rl_summative')

from stable_baselines3 import PPO
from environment.custom_env import SimpleAthleteNutritionEnv

def test_best_model():
    model = PPO.load('./models/ppo/run_0/final_model')
    env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
    
    obs, _ = env.reset()
    total_reward = 0
    day = 0
    
    print("Day | Action | Protein | Carbs | Reward | HRV | Fatigue | Glycogen")
    print("-" * 70)
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        protein, carbs = env._decode_action(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        day += 1
        
        print(f"{day:3d} | {action:6d} | {protein:7.1f} | {carbs:5.1f} | "
              f"{reward:6.1f} | {env.hrv:5.1f} | {env.fatigue:7.1f} | {env.glycogen:8.1f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward:.1f}")
    print(f"Expected: ~247.0")
    env.close()

if __name__ == "__main__":
    test_best_model()