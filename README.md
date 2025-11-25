# AI-Powered Athlete Nutrition Optimizer 🏋️‍♂️🧠

A reinforcement learning system that optimizes daily macronutrient profiles for athletes using four different RL algorithms. The agent learns to balance protein and carbohydrate intake to maximize performance and recovery across a 15-day training cycle.

**Video Recording:** https://www.youtube.com/watch?v=5viNXFT9OrY

---

## 📋 Project Overview

This project designs and implements an intelligent AI-Powered Athlete Nutrition Optimization System using Reinforcement Learning. The system trains intelligent agents to optimize daily macronutrient recommendations for a 70kg athlete across a 15-day training cycle.

The agent learns complex relationships between nutrition choices and physiological responses, maximizing sustained performance and recovery while minimizing injury risk. The project compares four RL algorithms from Stable-Baselines3:

- **DQN (Deep Q-Network)** - Value-based method
- **PPO (Proximal Policy Optimization)** - Policy gradient method  
- **A2C (Advantage Actor-Critic)** - Policy gradient method
- **REINFORCE** - Vanilla policy gradient method

Each algorithm is extensively tuned with 10 different hyperparameter configurations to ensure robust performance evaluation and fair comparison.

---

## 🎯 Problem Statement

Athletes require personalized daily nutrition plans that adapt to their current physiological state. Manual planning is time-consuming and often suboptimal. This system uses reinforcement learning to automatically adjust protein and carbohydrate intake to maximize recovery while managing fatigue during intensive training periods.

---

## 🏗️ Project Structure

```
student_name_rl_summative/
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment
│   ├── rendering.py           # Pygame visualization system
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
├── main.py                    # Entry point - run best model
├── run_random.py              # Random agent demonstration
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Gakwaya011/Christophe_Gakwaya_rl_summative.git
cd Christophe_Gakwaya_rl_summative
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p models/dqn models/ppo models/a2c results logs
```

### Basic Usage

**Visualize random agent (no training):**
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

**Run best performing model with visualization:**
```bash
PYTHONPATH=. python main.py
```

**Generate comprehensive analysis:**
```bash
PYTHONPATH=. python analysis/analyze_results.py
PYTHONPATH=. python analysis/training_plots.py
```

---

## 🎮 Environment Design

### State Space (Observation)

The agent observes four physiological metrics as a continuous state vector:

| Index | Metric | Range | Description |
|-------|--------|-------|-------------|
| 0 | Current Day | [0, 14] | Position in 15-day training cycle |
| 1 | HRV | [40, 100] ms | Heart Rate Variability - recovery indicator |
| 2 | Fatigue | [0, 100]% | Accumulated fatigue level |
| 3 | Glycogen | [0, 100]% | Muscle energy stores |

### Action Space

The action space is discrete with 9 actions representing all combinations of 3 protein levels and 3 carbohydrate levels:

| Protein Level | Low (1.2g/kg) | Medium (1.6g/kg) | High (2.0g/kg) |
|---|---|---|---|
| **Low Carbs (3.0g/kg)** | Action 0 | Action 1 | Action 2 |
| **Medium Carbs (5.0g/kg)** | Action 3 | Action 4 | Action 5 |
| **High Carbs (7.0g/kg)** | Action 6 | Action 7 | Action 8 |

**Total macronutrients for a 70kg athlete:**
- Protein: 84g (low), 112g (medium), 140g (high)
- Carbs: 210g (low), 350g (medium), 490g (high)

### Reward Function

The reward function encourages optimal physiological states with dense, informative feedback:

**Positive Rewards:**
- ✅ Optimal HRV (60-80 ms): +30
- ✅ Low fatigue (<40%): +20
- ✅ Optimal glycogen (60-90%): +15
- ✅ Positive protein balance: +10
- 🎯 Successful completion (15 days): +100
- 🎯 Partial completion: +50

**Negative Rewards:**
- ❌ High fatigue (>70%): -30
- ❌ Low glycogen (<30%): -20
- ❌ Sustained protein deficit (3+ days): -15
- ❌ Injury risk (fatigue >85%): -50
- ❌ Early termination (fatigue >95%): -100

### Termination Conditions

- Episode completes after 15 days of training
- Episode terminates early if fatigue exceeds 95% (injury threshold)

### Environment Visualization

The custom Pygame visualization provides real-time feedback with:
- Color-coded gauges: HRV (green), Fatigue (red), Glycogen (orange)
- 15-day progress timeline
- Current macronutrient recommendations
- Performance summary panel
- Episode statistics and rewards tracking

---

## 📊 Results Summary

### Algorithm Performance Comparison

| Algorithm | Mean Reward | Std Dev | Episodes to Converge | Stability |
|---|---|---|---|---|
| **PPO** | 247.0 ± 73.7 | ~100 | ~100 episodes | ⭐⭐⭐⭐⭐ |
| **A2C** | 226.5 ± 53.7 | ~150 | ~150 episodes | ⭐⭐⭐⭐ |
| **DQN** | 208.0 ± 139.3 | ~200 | ~200 episodes | ⭐⭐⭐ |
| **REINFORCE** | 61.5 ± 57.5 | >400 | >400 episodes | ⭐ |
| **Random Baseline** | -180 | — | — | — |

### Key Findings

- **PPO emerged as the superior algorithm** with highest mean reward (247.0), excellent stability (std=73.7), and fastest convergence (~100 episodes)
- **All learned algorithms significantly outperformed the random baseline** with improvements up to 427 points
- **A2C delivered respectable performance** (226.5 mean reward) with good stability, making it a practical alternative
- **DQN showed competitive performance** (208.0 mean reward) but suffered from higher variance and oscillatory convergence
- **REINFORCE struggled significantly** (61.5 mean reward), highlighting limitations of vanilla policy gradients in complex environments
- **PPO learned effective strategies:** High protein intake combined with intelligent carbohydrate cycling based on fatigue levels
- **Generalization:** PPO maintained performance within 15% across diverse initial conditions; REINFORCE showed poor generalization

---

## 🔧 Technical Implementation

### DQN (Deep Q-Network)

**Architecture:** 3-layer MLP with [256, 256] hidden units and ReLU activations

**Key Features:**
- Experience replay: Buffer sizes from 10K to 100K transitions
- Target network: Soft updates with Polyak averaging (τ=0.005)
- Double DQN: Reduced overestimation bias
- ϵ-greedy exploration: Linear decay from 1.0 to 0.01

**Hyperparameter Ranges Tested:**
- Learning rates: 1e-4 to 2e-3
- Discount factor (γ): 0.90 to 0.99
- Buffer size: 10K to 100K
- Batch size: 16 to 128

### PPO (Proximal Policy Optimization)

**Architecture:** Actor and Critic with shared [64, 64] base and separate heads

**Key Features:**
- Clipped objective function (ϵ=0.2) prevents destructive updates
- Multiple epochs per update (4-25 epochs)
- Advantage normalization for stable learning
- Mini-batch optimization

**Hyperparameter Ranges Tested:**
- Learning rates: 1e-4 to 1e-3
- n_steps: 512 to 4096
- Batch size: 16 to 128
- Epochs per update: 3 to 25

### A2C (Advantage Actor-Critic)

**Architecture:** Shared feature extraction with actor and critic heads

**Key Features:**
- Synchronous advantage actor-critic updates
- Generalized Advantage Estimation (GAE)
- Shared feature extraction
- Entropy regularization for exploration

**Hyperparameter Ranges Tested:**
- Learning rates: 5e-5 to 5e-4
- n_steps: 40 to 200
- GAE lambda (λ): 0.85 to 0.95
- Discount factor (γ): 0.93 to 0.99

### REINFORCE

**Architecture:** Single-layer network with [32, 512] hidden sizes

**Key Features:**
- Monte Carlo return estimation
- Baseline subtraction for variance reduction
- Direct policy gradient updates
- Reward normalization

**Hyperparameter Ranges Tested:**
- Learning rates: 1e-4 to 1e-2
- Discount factor (γ): 0.90 to 0.99
- Hidden sizes: 32 to 512

---

## 📈 Training Results Analysis

### Cumulative Rewards

**PPO** demonstrated the most efficient learning curve, achieving high rewards within ~100 episodes and maintaining stable performance throughout training. Quick convergence to ~250 reward points with minimal oscillation.

**A2C** displayed steady, consistent improvement with moderate sample efficiency. Smooth progression without dramatic oscillations, converging to performance between PPO and DQN.

**DQN** exhibited characteristic oscillatory behavior due to the moving target problem. Performance fluctuated during training but achieved competitive final performance (~200 reward points).

**REINFORCE** struggled significantly with high variance and slow convergence, showing dramatic reward oscillations throughout training, highlighting limitations of vanilla policy gradients.

### Training Stability

- **PPO's clipped objective function** demonstrated remarkable stability with smooth optimization curves
- **DQN's Q-network loss** showed oscillations from the moving target problem, but overall trend converged
- **Policy entropy analysis** revealed PPO and A2C maintain effective entropy reduction while REINFORCE keeps higher final entropy

### Convergence Analysis

- **PPO:** ~100 episodes (80% of max performance)
- **A2C:** ~150 episodes
- **DQN:** ~200 episodes
- **REINFORCE:** >400 episodes (inconsistent convergence)

### Generalization Testing

Trained models evaluated on unseen initial states with randomized physiological parameters:

- **PPO:** Most robust, maintaining performance within 15% of training baseline
- **A2C:** Reasonable generalization within 25% of training results
- **DQN:** Variable generalization, 20-40% performance variations
- **REINFORCE:** Poor generalization with highly inconsistent behavior

All algorithms significantly outperformed random baseline across all test conditions.

---

## 🎥 Visualization & Demonstration

The Pygame-based visualization provides:

- **Real-time monitoring** of all physiological metrics (HRV, Fatigue, Glycogen)
- **Interactive 15-day timeline** showing daily nutrition decisions and their effects
- **Current macronutrient display** with protein and carbohydrate amounts
- **Performance dashboard** showing cumulative rewards and episode statistics
- **Color-coded feedback** for intuitive understanding of physiological states

Run the best-performing agent visualization:
```bash
PYTHONPATH=. python main.py
```

---

## 📋 Implementation Notes

### Custom Environment (`custom_env.py`)

Implements the Gymnasium interface with:
- Realistic physiological dynamics simulating muscle recovery, fatigue accumulation, and glycogen depletion
- Deterministic state transitions with optional stochastic noise
- Comprehensive reward shaping for multi-objective optimization
- Edge case handling for extreme physiological states

### Rendering System (`rendering.py`)

Pygame-based visualization providing:
- Real-time display of agent state and decisions
- Interactive GUI for training and evaluation
- Episode statistics and performance metrics
- Support for recording demonstration videos
- Color-coded gauges and progress indicators

### Analysis Module

Provides comprehensive evaluation:
- Algorithm comparison across 10 hyperparameter configurations each
- Training curve generation and convergence analysis
- Stability metrics and variance analysis
- Generalization performance on unseen conditions
- Publication-quality visualization plots

---

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

---

## 💡 Algorithm Insights

### PPO's Strengths
- Stable policy updates through clipping mechanism
- Efficient sample utilization through multiple epochs
- Robust generalization to unseen states
- Best practical choice for real-world applications

### A2C's Advantages
- Simpler implementation than PPO
- Reasonable sample efficiency
- Stable learning without excessive computation

### DQN's Characteristics
- Effective for discrete action spaces
- Good final performance despite instability
- Challenges with moving target problem
- Exploration-exploitation balance sensitivity

### REINFORCE Limitations
- High-variance gradient estimates
- Poor sample efficiency
- Unreliable convergence
- Impractical for complex environments

---

## 🎓 Conclusions

PPO is the recommended algorithm for athlete nutrition optimization due to its superior stability, convergence speed, and generalization capabilities. The 427-point improvement over random baseline demonstrates that learned policies capture meaningful patterns in environment dynamics.

The project successfully demonstrates how reinforcement learning can optimize complex real-world problems involving multiple competing objectives (performance, recovery, injury prevention) across extended training cycles.

---

## 📚 References & Acknowledgments

- **Stable-Baselines3 Team:** For robust RL algorithm implementations
- **OpenAI Gymnasium:** For environment interface standardization
- **Pygame Community:** For visualization support
- **Course Instructors:** For guidance on RL principles and best practices

---

## 📄 License

This project is for educational purposes as part of the Machine Learning Techniques II Reinforcement Learning Summative Assignment at African Leadership University.

---

## 👨‍💻 Author

**Christophe Gakwaya**  
African Leadership University  
BSE - Machine Learning Techniques II  
Reinforcement Learning Summative Assignment

**GitHub:** https://github.com/Gakwaya011/Christophe_Gakwaya_rl_summative

---

## 📝 Additional Resources

- **Project Report:** See submitted PDF for detailed analysis
- **Video Demonstration:** https://www.youtube.com/watch?v=5viNXFT9OrY
- **Training Logs:** Located in `/logs` directory
- **Analysis Plots:** Located in `/results` directory
