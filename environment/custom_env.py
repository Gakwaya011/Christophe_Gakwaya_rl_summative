import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from .rendering import PygameRenderer

# --- Constants for Simulation ---
# Defines the effect of each action (diet) on the athlete's recovery potential.
# Higher value means a greater positive impact on recovery metrics (HRV, Fatigue)
RECOVERY_EFFECTS = {
    0: 1.5,  # Recovery (High Protein, Low Carb) - Maximize repair
    1: 1.0,  # Endurance (High Carb) - Good for energy
    2: 0.5,  # Balanced Maintenance - Neutral
    3: -0.5, # Calorie Surplus - Focus on mass, may slightly delay recovery rate
    4: -1.0, # Calorie Deficit - Stressful on the body, actively hinders recovery
}

# --- Main Environment Class ---
class AthleteRecoveryEnv(gym.Env):
    """
    A custom Gymnasium environment for an RL agent (Nutrition Recommender)
    to optimize an athlete's recovery and performance.
    """
    
    # 1. Initialization and Space Definition
    def __init__(self, max_ep_steps=30):
        super().__init__()
        self.max_ep_steps = max_ep_steps
        self.current_step = 0

        # Action Space: Discrete set of 5 diet profiles (the agent's recommendation)
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: 5 key metrics describing the athlete's state
        # [Recovery_HRV, Fatigue_Score, Energy_Deficit, Workout_Intensity, Days_in_Phase]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1000.0, 0.0, 0]),
            high=np.array([1.0, 5.0, 1000.0, 1.0, 30]),
            dtype=np.float32
        )
        
        # Map for Action Index to descriptive name (used in R_alignment calculation)
        self.ACTION_MAP = {
            0: 'Recovery', 1: 'Endurance', 2: 'Balanced', 3: 'Surplus', 4: 'Deficit'
        }
        
        # Internal state variable
        self.state = None

    # 2. Reset Function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize the athlete to a 'baseline/recovered' state
        initial_hrv = random.uniform(0.8, 0.9) # High Recovery
        initial_fatigue = random.uniform(0.5, 1.0) # Low Fatigue
        initial_deficit = 0.0 # Balanced Energy
        initial_intensity = random.uniform(0.1, 0.3) # Low initial intensity
        initial_days = 1
        
        self.state = np.array([
            initial_hrv, 
            initial_fatigue, 
            initial_deficit, 
            initial_intensity, 
            initial_days
        ], dtype=np.float32)
        
        # Placeholder for auxiliary information
        info = {}
        
        return self.state, info

    # 3. Step Function (The Core RL Loop)
    def step(self, action):
        self.current_step += 1
        
        # --- 1. Apply Action and Simulate State Transition ---
        
        # Get the effect of the recommended diet
        diet_effect = RECOVERY_EFFECTS[action]
        
        # Current State Variables (for cleaner reading)
        hrv, fatigue, deficit, intensity, days = self.state
        
        # Simulate New State (Day t+1) based on Action and Current State
        
        # A. New Recovery (HRV): Improved by diet effect, reduced by previous intensity/fatigue
        new_hrv = hrv + (diet_effect * 0.1) - (fatigue * 0.05) + random.uniform(-0.02, 0.02)
        
        # B. New Fatigue: Increased by previous intensity, reduced by good recovery action
        new_fatigue = fatigue + (intensity * 0.4) - (diet_effect * 0.15) + random.uniform(-0.1, 0.1)
        
        # C. New Energy Deficit: Directly linked to diet profile (Surplus/Deficit)
        # Assuming high carb/surplus adds energy, deficit subtracts
        energy_change = [50, 100, 0, 200, -200][action] 
        new_deficit = deficit + energy_change
        
        # D. Simulate Next Day's Workout Intensity (The Outcome Metric)
        # Intensity should be hampered by high fatigue but boosted by high recovery (HRV)
        base_intensity_potential = 0.6 # The athlete's goal intensity
        potential_loss = new_fatigue * 0.1 # Penalty from fatigue
        potential_gain = new_hrv * 0.2 # Gain from recovery
        new_intensity = base_intensity_potential + potential_gain - potential_loss + random.uniform(-0.1, 0.1)
        
        # Clip all new state values to their bounds
        new_state = np.array([
            np.clip(new_hrv, 0.0, 1.0),
            np.clip(new_fatigue, 0.0, 5.0),
            np.clip(new_deficit, -1000.0, 1000.0),
            np.clip(new_intensity, 0.0, 1.0), # Intensity cannot be negative or > 1.0
            days + 1
        ], dtype=np.float32)
        
        # --- 2. Calculate Reward ---
        
        # Reward 1: Recovery Bonus (Agent's primary goal)
        R_recovery = 50.0 * new_state[0] # High HRV is rewarded heavily
        
        # Reward 2: Fatigue/Injury Penalty
        R_fatigue = -20.0 * new_state[1] # High fatigue is penalized
        
        # Reward 3: Performance Alignment (Rewarding sustainable performance)
        # Target: Keep intensity high (near 0.6-0.8) without crashing recovery metrics.
        # This rewards the agent for not letting performance drop, and penalizes crashes.
        R_performance_alignment = 50.0 * new_state[3] # Reward based on next day's achieved intensity
        
        # Reward 4: Penalty for extreme states (e.g., massive deficit)
        R_stability_penalty = 0.0
        if abs(new_deficit) > 800:
             R_stability_penalty = -50.0 # Punish unsustainable energy management
        
        reward = R_recovery + R_fatigue + R_performance_alignment + R_stability_penalty

        # --- 3. Check Termination Conditions ---
        
        terminated = False
        truncated = False
        
        # Termination: Athlete crashes (Injury/Burnout) - severe fatigue
        if new_fatigue >= 4.5:
            terminated = True
            reward -= 200 # Heavy penalty for failing the mission
            
        # Truncation: Max steps reached (Episode Time Limit)
        if self.current_step >= self.max_ep_steps:
            truncated = True

        self.state = new_state
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        # This is where your pygame/visualization code will go.
        # For now, we'll keep it simple until the visualization file is created.
        pass

    def close(self):
        # Used for cleaning up resources, like closing the pygame window.
        pass

# End of custom_env.py