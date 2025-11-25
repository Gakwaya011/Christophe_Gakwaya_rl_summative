import sys
import time
import pygame
sys.path.append('.')

from environment.custom_env import SimpleAthleteNutritionEnv
from environment.rendering import NutritionVisualizer

def main():
    """Run random agent demonstration"""
    print("Random Agent Demonstration")
    
    env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
    visualizer = NutritionVisualizer()
    
    observation, info = env.reset()
    
    def get_viz_state():
        return {
            'current_day': env.current_day,
            'hrv': env.hrv,
            'fatigue': env.fatigue,
            'glycogen': env.glycogen,
            'history': env.history
        }
    
    episode_reward = 0
    done = False
    
    print("Day | Action | Protein | Carbs | Reward | HRV | Fatigue | Glycogen")
    print("-" * 70)
    
    while not done:
        action = env.action_space.sample()
        protein, carbs = env._decode_action(action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        print(f"{env.current_day:3d} | {action:6d} | {protein:7.1f} | {carbs:5.1f} | "
              f"{reward:6.1f} | {env.hrv:5.1f} | {env.fatigue:7.1f} | {env.glycogen:8.1f}")
        
        viz_state = get_viz_state()
        visualizer.render(viz_state, (protein, carbs))
        time.sleep(0.3)
        
        if done:
            print(f"\nTotal Reward: {episode_reward:.2f}")
            print("Close window to exit.")
            while True:
                viz_state = get_viz_state()
                visualizer.render(viz_state, (protein, carbs))
    
    visualizer.close()
    env.close()

if __name__ == "__main__":
    main()