import os
import sys
import time
import pandas as pd

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import SimpleAthleteNutritionEnv
from environment.rendering import NutritionVisualizer

def find_best_model():
    """Find the best performing model across all algorithms"""
    best_score = -float('inf')
    best_model_path = None
    best_algo = None
    
    algorithms = {
        'dqn': DQN,
        'ppo': PPO, 
        'a2c': A2C
    }
    
    for algo, model_class in algorithms.items():
        results_path = f'./results/{algo}_results.csv'
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            best_run = results_df.loc[results_df['mean_reward'].idxmax()]
            
            if best_run['mean_reward'] > best_score:
                best_score = best_run['mean_reward']
                best_model_path = f"./models/{algo}/run_{int(best_run['run_id'])}/final_model"
                best_algo = algo
    
    return best_model_path, best_algo, best_score

def run_best_model():
    """Run the best performing model with visualization"""
    
    # Find the best model automatically
    model_path, algo, expected_score = find_best_model()
    
    if model_path is None or not os.path.exists(model_path + ".zip"):
        print("No trained model found. Please train models first.")
        print("Falling back to PPO Run 0...")
        model_path = "./models/ppo/run_0/final_model"
        algo = "PPO"
        expected_score = 247.0
    
    print(f"🚀 Loading best model: {algo.upper()} (Expected: {expected_score:.1f})")
    print(f"📁 Model path: {model_path}")
    
    # Load model
    if algo == 'dqn':
        model = DQN.load(model_path)
    elif algo == 'ppo':
        model = PPO.load(model_path)
    elif algo == 'a2c':
        model = A2C.load(model_path)
    else:
        print("Model type not supported")
        return
    
    # Create environment and visualizer
    env = SimpleAthleteNutritionEnv(athlete_weight=70.0)
    visualizer = NutritionVisualizer()
    
    # Run episode
    observation, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    print("\n🎮 Running best model...")
    print("❌ Close visualization window to stop")
    print("\nDay | Action | Protein | Carbs | Step Reward | Total Reward | HRV | Fatigue | Glycogen")
    print("-" * 90)
    
    while not done:
        # Get action from model
        action, _states = model.predict(observation, deterministic=True)
        
        # Take step
        observation, step_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += step_reward
        step_count += 1
        
        # Decode action for visualization
        protein, carbs = env._decode_action(action)
        
        # Prepare state for visualization
        viz_state = {
            'current_day': env.current_day,
            'hrv': env.hrv,
            'fatigue': env.fatigue,
            'glycogen': env.glycogen,
            'history': env.history
        }
        
        # Print step info
        print(f"{env.current_day:3d} | {action:6d} | {protein:7.1f} | {carbs:5.1f} | "
              f"{step_reward:11.1f} | {total_reward:12.1f} | {env.hrv:5.1f} | {env.fatigue:7.1f} | {env.glycogen:8.1f}")
        
        # Render
        visualizer.render(viz_state, (protein, carbs))
        
        # Slow down for viewing
        time.sleep(0.3)
        
        if done:
            print(f"\n{'='*50}")
            print(f"🎉 EPISODE COMPLETE!")
            print(f"{'='*50}")
            print(f"📊 Performance Summary:")
            print(f"   Total Days: {step_count}")
            print(f"   Total Reward: {total_reward:.1f}")
            print(f"   Expected Range: {expected_score:.1f}")
            print(f"   Final HRV: {env.hrv:.1f} ms")
            print(f"   Final Fatigue: {env.fatigue:.1f}%")
            print(f"   Final Glycogen: {env.glycogen:.1f}%")
            
            # Performance assessment
            if total_reward > expected_score + 50:
                print(f"   ✅ OUTSTANDING: Above expectations!")
            elif total_reward >= expected_score:
                print(f"   ✅ EXCELLENT: Met expectations!")
            else:
                print(f"   ⚠️  GOOD: Within expected range")
            
            print(f"{'='*50}")
            
            # Keep window open
            print("🖥️  Close visualization window to exit.")
            while True:
                visualizer.render(viz_state, (protein, carbs))
    
    visualizer.close()
    env.close()

if __name__ == "__main__":
    run_best_model()