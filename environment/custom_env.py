import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

class SimpleAthleteNutritionEnv(gym.Env):
    """
    Simplified Environment for Athlete Nutrition Optimization
    
    Core Mission: Learn to balance 3 key macronutrients for optimal recovery
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, athlete_weight: float = 70.0, render_mode: Optional[str] = None):
        super().__init__()
        
        self.athlete_weight = athlete_weight
        self.render_mode = render_mode
        
        # SIMPLIFIED: 9 discrete actions (3 protein levels × 3 carb levels)
        self.action_space = spaces.Discrete(9)
        
        # SIMPLIFIED: 4 core features instead of 8
        self.observation_space = spaces.Box(
            low=np.array([0, 40, 0, 0], dtype=np.float32),    # day, hrv, fatigue, glycogen
            high=np.array([14, 100, 100, 100], dtype=np.float32),  # 15-day episodes
            dtype=np.float32
        )
        
        # SIMPLIFIED: Fewer macro levels
        self.protein_levels = np.array([1.2, 1.6, 2.0])  # g/kg
        self.carb_levels = np.array([3.0, 5.0, 7.0])     # g/kg
        
        # Training schedule (simpler)
        self.training_schedule = self._generate_training_schedule()
        
        # State variables
        self.current_day = 0
        self.hrv = 70.0
        self.fatigue = 30.0
        self.glycogen = 80.0
        
        # History for visualization
        self.history = {'hrv': [], 'fatigue': [], 'glycogen': [], 'rewards': []}
        
    def _generate_training_schedule(self) -> np.ndarray:
        """Simplified 15-day schedule"""
        schedule = np.zeros(15)
        for week in range(2):
            base = week * 7
            schedule[base:base+5] = np.random.uniform(6, 8, 5)  # Training
            schedule[base+5] = 3  # Light
            schedule[base+6] = 0  # Rest
        schedule[-1] = 0  # Final rest day
        return schedule
    
    def _decode_action(self, action: int) -> Tuple[float, float]:
        """Convert 9 discrete actions to protein/carb values"""
        protein_idx = action // 3
        carb_idx = action % 3
        
        protein = self.protein_levels[protein_idx] * self.athlete_weight
        carbs = self.carb_levels[carb_idx] * self.athlete_weight
        
        return protein, carbs
    
    def _update_physiology(self, protein: float, carbs: float):
        """SIMPLIFIED physiological model"""
        training = self.training_schedule[self.current_day]
        
        # 1. Glycogen dynamics (simplified)
        glycogen_use = training * 7
        glycogen_gain = carbs * 0.12
        self.glycogen = np.clip(self.glycogen - glycogen_use + glycogen_gain, 0, 100)
        
        # 2. Fatigue (simplified)
        fatigue_gain = training * 5
        recovery = (protein / (1.6 * self.athlete_weight)) * 12 + (self.glycogen / 100) * 8
        self.fatigue = np.clip(self.fatigue + fatigue_gain - recovery, 0, 100)
        
        # 3. HRV (simplified)
        hrv_change = 0
        if self.fatigue < 40: hrv_change += 1.5
        elif self.fatigue > 70: hrv_change -= 2
        
        if self.glycogen > 60: hrv_change += 1
        
        # Add small randomness
        hrv_change += np.random.normal(0, 1)
        self.hrv = np.clip(self.hrv + hrv_change, 40, 100)
    
    def _calculate_reward(self) -> float:
        """SIMPLIFIED reward function with denser rewards"""
        reward = 0
        
        # HRV reward (denser)
        if 60 <= self.hrv <= 80:
            reward += 25
        elif 55 <= self.hrv < 60 or 80 < self.hrv <= 85:
            reward += 10
        elif self.hrv < 50:
            reward -= 15
        
        # Fatigue reward (denser)
        if self.fatigue < 40:
            reward += 20
        elif self.fatigue > 70:
            reward -= 20
        
        # Glycogen reward (denser)
        if 60 <= self.glycogen <= 90:
            reward += 15
        elif self.glycogen < 30:
            reward -= 15
        
        # Small penalty for extreme values (encourages balance)
        if self.fatigue > 85:
            reward -= 25
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset state
        self.current_day = 0
        self.hrv = 70.0 + np.random.uniform(-5, 5)
        self.fatigue = 30.0 + np.random.uniform(-10, 10)
        self.glycogen = 80.0 + np.random.uniform(-10, 10)
        
        self.history = {'hrv': [], 'fatigue': [], 'glycogen': [], 'rewards': []}
        self.training_schedule = self._generate_training_schedule()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        protein, carbs = self._decode_action(action)
        
        # Update state
        self._update_physiology(protein, carbs)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Store history
        self.history['hrv'].append(self.hrv)
        self.history['fatigue'].append(self.fatigue)
        self.history['glycogen'].append(self.glycogen)
        self.history['rewards'].append(reward)
        
        # Advance day
        self.current_day += 1
        
        # Check termination (15 days instead of 30)
        terminated = self.current_day >= 14
        
        # Terminal bonus
        if terminated:
            avg_hrv = np.mean(self.history['hrv'])
            if avg_hrv > 65:
                reward += 100
            elif avg_hrv > 60:
                reward += 50
        
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.current_day,
            self.hrv,
            self.fatigue,
            self.glycogen
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'day': self.current_day,
            'training_today': self.training_schedule[self.current_day] if self.current_day < 15 else 0,
            'total_reward': sum(self.history['rewards']) if self.history['rewards'] else 0
        }
    def render(self):
        if self.render_mode == "human":
            total_reward = sum(self.history['rewards']) if self.history['rewards'] else 0
            print(f"\n=== Day {self.current_day + 1}/30 ===")
            print(f"HRV: {self.hrv:.1f} ms | Fatigue: {self.fatigue:.1f}% | Glycogen: {self.glycogen:.1f}%")
            print(f"Protein Balance: {self.protein_balance:+.1f}g | Hydration: {self.hydration:.1f}%")
            print(f"Training Load: {self.training_load:.1f} | Weight Change: {self.weight_change:+.2f}kg")
            print(f"Step Reward: {self.history['rewards'][-1] if self.history['rewards'] else 0:.1f} | Total Reward: {total_reward:.1f}")
    
