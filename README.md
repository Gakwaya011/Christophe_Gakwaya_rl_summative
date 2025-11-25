# AI-Powered Athlete Nutrition Optimizer 🏋️‍♂️🧠

A reinforcement learning system that optimizes daily macronutrient profiles for athletes using four different RL algorithms. The agent learns to balance protein and carbohydrate intake to maximize performance and recovery across a 15-day training cycle.

## 📋 Project Overview

This project implements and compares four reinforcement learning algorithms for athlete nutrition optimization:

- **DQN (Deep Q-Network)** - Value-based method
- **PPO (Proximal Policy Optimization)** - Policy gradient method
- **A2C (Advantage Actor-Critic)** - Policy gradient method
- **REINFORCE** - Vanilla policy gradient method

The system trains agents to make optimal daily nutrition decisions based on physiological metrics including Heart Rate Variability (HRV), fatigue levels, and glycogen stores. This addresses the real-world challenge of personalizing athlete nutrition plans to maximize performance while managing fatigue and recovery.

## 🏗️ Project Structure

```
student_name_rl_summative/
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment
│   ├── rendering.py           # Pygame visualization
│   └── __init__.py
├── training/
│   ├── dqn_training.py        # DQN training with 10 hyperparameter runs
│   ├── pg_training.py         # PPO & A2C training
│   ├── reinforce_training.py  # REINFORCE training
│   └── __init__.py
├── analysis/
│   ├── analyze_results.py     # Results analysis and comparison
│   ├── training_plots.py      # Training visualization generation
│   └── __init__.py
├── models/
│   ├── dqn/                   # Saved DQN models
│   └── pg/                    # Saved policy gradient models
├── results/                   # Training results and plots
├── main.py                    # Run best performing model
├── run_random.py              # Random agent demonstration
├── requirements.txt           # Project dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student_name_rl_summative.git
cd student_name_rl_summative
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up directories:
```bash
mkdir -p models/dqn models/ppo models/a2c results logs
```

### Basic Usage

**Run random agent demonstration (visualization only, no training):**
```bash
PYTHONPATH=. python run_random.py
```

**Train all algorithms (10 runs each with different hyperparameters):**
```bash
# Train DQN
PYTHONPATH=. python training/dqn_training.py

# Train PPO & A2C  
PYTHONPATH=. python training/pg_training.py

# Train REINFORCE
PYTHONPATH=. python training/reinforce_training.py
```

**Run the best performing model:**
```bash
PYTHONPATH=. python main.py
```

**Analyze results:**
```bash
PYTHONPATH=. python analysis/analyze_results.py
PYTHONPATH=. python analysis/training_plots.py
```

## 🎯 Environment Details

### Problem Statement

Athletes need personalized daily nutrition plans that optimize macronutrient intake based on their physiological state. Manual planning is time-consuming and often suboptimal. This system uses reinforcement learning to automatically adjust protein and carbohydrate intake to maximize recovery while managing fatigue during a 15-day training cycle.

### State Space (Observation)

The agent observes four physiological metrics:

- **Current Day** (0-14): Position in training cycle
- **HRV** (40-100 ms): Heart Rate Variability indicating recovery status
- **Fatigue** (0-100%): Accumulated fatigue level
- **Glycogen** (0-100%): Muscle energy stores

### Action Space

9 discrete actions representing protein-carbohydrate combinations:

| Protein Level | Low (1.2g/kg) | Medium (1.6g/kg) | High (2.0g/kg) |
|---|---|---|---|
| **Low Carbs (3.0g/kg)** | Action 0 | Action 1 | Action 2 |
| **Medium Carbs (5.0g/kg)** | Action 3 | Action 4 | Action 5 |
| **High Carbs (7.0g/kg)** | Action 6 | Action 7 | Action 8 |

### Reward Function

The agent receives rewards for maintaining optimal physiological states:

- ✅ Optimal HRV (60-80 ms): +30
- ✅ Low fatigue (<40%): +20
- ✅ Optimal glycogen (60-90%): +15
- ✅ Positive protein balance: +10
- ❌ High fatigue (>70%): -30
- ❌ Injury risk (fatigue >85%): -50
- 🎯 Terminal bonus for successful completion: +100 (success) / +50 (partial)

### Termination Conditions

Episode terminates after 15 days or if fatigue exceeds 95% (injury threshold).

## 📊 Results Summary

### Algorithm Performance Ranking

| Algorithm | Mean Reward | Std Dev | Convergence | Stability |
|---|---|---|---|---|
| **PPO** | 247.0 ± 73.7 | ~100 | ~200 episodes | ⭐⭐⭐⭐⭐ |
| **A2C** | 226.5 ± 53.7 | ~150 | ~250 episodes | ⭐⭐⭐⭐ |
| **DQN** | 208.0 ± 139.3 | ~200 | ~300 episodes | ⭐⭐⭐ |
| **REINFORCE** | 61.5 ± 57.5 | >400 | >400 episodes | ⭐ |
| **Random Baseline** | -180 | — | — | — |

### Key Findings

- PPO achieved the best performance with stable training and fastest convergence
- All learned algorithms significantly outperformed random baseline
- Best improvement: 427 points over random selection
- PPO learned effective strategies: High protein intake combined with intelligent carbohydrate cycling based on fatigue levels
- A2C demonstrated good stability with slightly lower performance than PPO

## 🎥 Visualization

The project includes a Pygame-based visualization showing:

- Real-time physiological metrics (HRV, Fatigue, Glycogen)
- 15-day progress timeline with daily nutrition decisions
- Current macronutrient profile being executed
- Performance summary and episode statistics

Run the visualization with your best trained agent:
```bash
PYTHONPATH=. python main.py
```

## 🔧 Technical Details

### Algorithms Implemented

**DQN (Deep Q-Network):**
- Experience replay buffer with prioritization
- Target network for stability
- ϵ-greedy exploration strategy
- Hyperparameter ranges: LR [1e-4, 5e-4], discount [0.95, 0.99], buffer size [1000, 50000]

**PPO (Proximal Policy Optimization):**
- Clipped objective function for stable updates
- Multiple epochs per update cycle
- Advantage normalization and value function scaling
- Hyperparameter ranges: LR [1e-4, 1e-3], clip range [0.1, 0.3], n_steps [256, 2048]

**A2C (Advantage Actor-Critic):**
- Synchronous advantage estimation
- Shared feature extraction for actor and critic
- Entropy regularization for exploration
- Hyperparameter ranges: LR [5e-5, 5e-4], n_steps [64, 512], ent_coef [0.0, 0.01]

**REINFORCE:**
- Monte Carlo return estimation
- Baseline subtraction for variance reduction
- Direct policy gradient updates
- Hyperparameter ranges: LR [1e-4, 1e-2], discount [0.9, 0.99]

### Hyperparameter Tuning

Each algorithm was tested with 10 different hyperparameter configurations to identify optimal settings. Key parameters tuned include:

- Learning rates: 1e-4 to 2e-3
- Discount factors: 0.9 to 0.99
- Network architectures: Hidden layer sizes [64, 128, 256]
- Exploration strategies: Different schedules and epsilon decay rates
- Update frequencies and batch sizes

## 📈 Analysis Features

The analysis module provides comprehensive evaluation across all algorithms:

- Algorithm comparison across 10 hyperparameter configurations each
- Training curves showing convergence behavior
- Stability metrics and variance analysis
- Generalization testing on unseen physiological conditions
- Performance visualization with professional publication-quality plots

Generate comprehensive analysis reports:
```bash
PYTHONPATH=. python analysis/analyze_results.py
```

## 🛠️ Requirements

```
gymnasium==0.29.1
stable-baselines3==2.0.0
pygame==2.5.2
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
torch==2.1.0
```

## 📋 Implementation Notes

### Custom Environment
The custom environment (`custom_env.py`) implements the Gymnasium interface with:
- Realistic physiological dynamics simulating muscle recovery, fatigue accumulation, and glycogen depletion
- Deterministic state transitions with optional stochastic noise
- Comprehensive reward shaping for multi-objective optimization

### Rendering System
The Pygame visualization (`rendering.py`) provides:
- Real-time display of agent state and decisions
- Interactive GUI for training and evaluation
- Episode statistics and performance metrics
- Support for recording demonstration videos

## 📝 Project Report

The comprehensive report includes:

- Environment design rationale and physiological modeling
- Detailed algorithm implementation and hyperparameter analysis
- Training stability and convergence analysis across all methods
- Generalization performance evaluation
- Comparative performance metrics and visual analysis
- Conclusions and recommendations for future improvements

## 📄 License

This project is for educational purposes as part of the Reinforcement Learning summative assignment.

## 👨‍💻 Author

[Your Name]  
Reinforcement Learning Summative Assignment

## 🙏 Acknowledgments

- Stable-Baselines3 team for robust RL algorithm implementations
- OpenAI Gymnasium for the environment interface standard
- Pygame community for visualization support
- Course instructors for guidance on RL principles and best practices