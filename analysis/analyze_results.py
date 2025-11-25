import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_all_results():
    """Load results from all algorithms"""
    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    results = []
    
    for algo in algorithms:
        try:
            df = pd.read_csv(f'./results/{algo}_results.csv')
            results.append(df)
        except FileNotFoundError:
            print(f"Results for {algo} not found")
    
    return pd.concat(results, ignore_index=True)

def plot_comparison():
    """Create comparison plots"""
    results = load_all_results()
    
    plt.figure(figsize=(12, 8))
    
    # Mean reward comparison
    plt.subplot(2, 2, 1)
    sns.boxplot(data=results, x='algorithm', y='mean_reward')
    plt.title('Mean Reward by Algorithm')
    plt.xticks(rotation=45)
    
    # Standard deviation comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(data=results, x='algorithm', y='std_reward')
    plt.title('Reward Std Dev by Algorithm')
    plt.xticks(rotation=45)
    
    # Training curves
    plt.subplot(2, 2, 3)
    for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
        try:
            curves = pd.read_csv(f'./results/{algo}_training_curves.csv')
            # Smooth the curves
            smoothed = curves.groupby(['algorithm', 'episode'])['reward'].mean().reset_index()
            plt.plot(smoothed['episode'], smoothed['reward'], label=algo.upper(), linewidth=2)
        except FileNotFoundError:
            continue
    
    plt.title('Training Curves (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Hyperparameter analysis
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=results, x='learning_rate', y='mean_reward', hue='algorithm', style='algorithm')
    plt.xscale('log')
    plt.title('Learning Rate vs Performance')
    
    plt.tight_layout()
    plt.savefig('./results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_best_performers():
    """Print best performing configurations"""
    results = load_all_results()
    
    print("BEST PERFORMERS BY ALGORITHM:")
    print("="*50)
    
    for algo in results['algorithm'].unique():
        algo_results = results[results['algorithm'] == algo]
        best = algo_results.loc[algo_results['mean_reward'].idxmax()]
        
        print(f"\n{algo.upper()}:")
        print(f"  Mean Reward: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
        print(f"  Run ID: {best['run_id']}")
        
        # Print hyperparameters
        hp_cols = [col for col in best.index if col not in ['run_id', 'algorithm', 'mean_reward', 'std_reward']]
        for hp in hp_cols:
            if pd.notna(best[hp]):
                print(f"  {hp}: {best[hp]}")

if __name__ == "__main__":
    os.makedirs('./results', exist_ok=True)
    plot_comparison()
    print_best_performers()