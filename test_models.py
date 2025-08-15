import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

from rl_environment import SmartBuildingEnv
from rl_agents import DQNAgent, PPOAgent, A3CAgent
from data_manager import DataManager

class ModelTester:
    """Comprehensive testing framework for evaluating trained RL models."""
    
    def __init__(self, models_dir: str = 'models', data_dir: str = 'citylearn_dataset'):
        """Initialize the model tester."""
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load best models for each algorithm
        try:
            self.best_models = self._load_best_models()
        except Exception as e:
            print(f"\nError loading models: {str(e)}")
            self.best_models = {}
        
        try:
            # Create environment
            self.env = SmartBuildingEnv(
                data_dir=data_dir,
                building_ids=None,  # Use all buildings
                time_step_minutes=15,
                episode_hours=24
            )
        except Exception as e:
            print(f"\nError creating environment: {str(e)}")
            raise
        
        # Store results
        self.results = {}
        self.current_results = {}  # Store results for single algorithm testing
        
    def _load_best_models(self) -> Dict:
        """Load the best saved models for each algorithm."""
        best_models = {}
        algorithms = ['dqn', 'ppo', 'a3c']
        
        for algo in algorithms:
            # Look for best model config file
            config_file = f'{algo}_config_best.json'
            config_path = os.path.join(self.models_dir, config_file)
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Verify model file exists
                    if not os.path.exists(config['model_path']):
                        print(f"Warning: Model file not found at {config['model_path']} for {algo.upper()}")
                        continue
                        
                    best_models[algo] = {
                        'config': config,
                        'model_path': config['model_path']
                    }
                    print(f"Loaded best {algo.upper()} model from {config_path}")
                except Exception as e:
                    print(f"Warning: Error loading {algo.upper()} model configuration: {str(e)}")
            else:
                print(f"Warning: No best model configuration found for {algo.upper()} at {config_path}")
                
        return best_models
        
    def _create_agent(self, algorithm: str) -> object:
        """Create and load a trained agent."""
        if algorithm not in self.best_models:
            print(f"\nNo trained model found for {algorithm.upper()}. Please train the model first.")
            return None
            
        try:
            # Get model path
            model_path = self.best_models[algorithm]['model_path']
            
            # Create agent based on algorithm
            if algorithm == 'dqn':
                agent = DQNAgent(
                    state_dims={'building': 8 * len(self.env.building_ids)},
                    action_dims={f'building_{b}': self.env.action_space[f'building_{b}'].n 
                               for b in self.env.building_ids}
                )
                agent.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
                agent.policy_net.eval()
                
            elif algorithm == 'ppo':
                n_actions = len(self.env.temp_setpoints) * len(self.env.building_ids)
                agent = PPOAgent(
                    state_dims={'building': 8 * len(self.env.building_ids)},
                    n_actions=n_actions
                )
                agent.network.load_state_dict(torch.load(model_path, weights_only=True))
                agent.network.eval()
                
            else:  # a3c
                n_actions = len(self.env.temp_setpoints) * len(self.env.building_ids)
                agent = A3CAgent(
                    state_dims={'building': 8 * len(self.env.building_ids)},
                    n_actions=n_actions,
                    env_creator=lambda: None,  # Not needed for evaluation
                    n_workers=1  # Single worker for evaluation
                )
                agent.global_network.load_state_dict(torch.load(model_path, weights_only=True))
                agent.global_network.eval()
                
            return agent
            
        except Exception as e:
            print(f"\nError creating {algorithm.upper()} agent: {str(e)}")
            return None
            
    def _process_action(self, algorithm: str, raw_action, building_ids: List[int]) -> Dict[str, int]:
        """Convert raw action from agent to environment-compatible format."""
        if algorithm == 'dqn':
            # DQN already returns dict format
            return raw_action
        else:  # ppo or a3c
            # For PPO/A3C, convert single action index to per-building actions
            if isinstance(raw_action, tuple):
                # PPO returns (action, value, policy)
                action_idx = raw_action[0]
            else:
                action_idx = raw_action
                
            # Calculate actions for each building
            n_temp_setpoints = len(self.env.temp_setpoints)
            actions = {}
            for i, b_id in enumerate(building_ids):
                building_action = action_idx % n_temp_setpoints
                action_idx //= n_temp_setpoints
                actions[f'building_{b_id}'] = building_action
            return actions

    def run_scenario_test(self, algorithm: str, scenario: str, n_episodes: int = 10) -> Dict:
        """Run scenario-based testing."""
        # Create agent
        agent = self._create_agent(algorithm)
        if agent is None:
            return None
            
        rewards = []
        energy_consumptions = []
        comfort_violations = []
        
        try:
            for episode in range(n_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                episode_energy = 0
                episode_violations = 0
                done = False
                
                while not done:
                    # Select action and convert to environment format
                    raw_action = agent.select_action(state, training=False)
                    action = self._process_action(algorithm, raw_action, self.env.building_ids)
                    
                    # Take step in environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Track metrics
                    episode_reward += reward
                    for building_info in info['buildings'].values():
                        episode_energy += building_info['energy_consumption']
                        if abs(building_info['comfort_penalty']) > 0:
                            episode_violations += 1
                            
                    state = next_state
                    
                rewards.append(episode_reward)
                energy_consumptions.append(episode_energy)
                comfort_violations.append(episode_violations)
                
            # Store results for single algorithm testing
            if algorithm not in self.current_results:
                self.current_results[algorithm] = {}
                
            results = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'mean_energy': float(np.mean(energy_consumptions)),
                'mean_violations': float(np.mean(comfort_violations))
            }
            
            self.current_results[algorithm][scenario] = results
            return results
            
        except Exception as e:
            print(f"\nError during scenario testing: {str(e)}")
            return None
        
    def compare_algorithms(self, n_episodes: int = 10) -> Dict:
        """Compare performance across all algorithms."""
        scenarios = ['summer', 'winter', 'weekday', 'weekend']
        algorithms = ['dqn', 'ppo', 'a3c']
        results = {}
        
        for algo in algorithms:
            algo_results = {}
            for scenario in scenarios:
                print(f"\nTesting {algo.upper()} in {scenario} scenario...")
                scenario_results = self.run_scenario_test(algo, scenario, n_episodes)
                algo_results[scenario] = scenario_results
            results[algo] = algo_results
            
        # Store results
        self.results = results
        return results
        
    def get_comparison_results(self) -> Dict:
        """Get results for algorithm comparison."""
        if not self.current_results:
            raise ValueError("No results available. Run testing first.")
        return self.current_results

    def get_current_results(self) -> Dict:
        """Get results for single algorithm testing."""
        if not self.current_results:
            raise ValueError("No results available. Run testing first.")
        return list(self.current_results.values())[0]  # Return first (and only) algorithm's results

    def statistical_analysis(self) -> Dict:
        """Perform statistical analysis on results."""
        if not self.current_results:
            raise ValueError("No results available. Run testing first.")
            
        analysis = {}
        algorithms = list(self.current_results.keys())
        
        # Perform t-tests between algorithms
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                # Compare mean rewards across all scenarios
                algo1_rewards = []
                algo2_rewards = []
                
                for scenario in self.current_results[algo1].keys():
                    algo1_rewards.extend(self.current_results[algo1][scenario]['rewards'])
                    algo2_rewards.extend(self.current_results[algo2][scenario]['rewards'])
                    
                t_stat, p_value = stats.ttest_ind(algo1_rewards, algo2_rewards)
                analysis[f'{algo1}_vs_{algo2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
        return analysis
        
    def _convert_to_native_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [self._convert_to_native_types(x) for x in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def plot_results(self, save_dir: str = 'results'):
        """Generate visualization plots."""
        # Use current results for plotting
        plot_data = self.current_results
        
        if not plot_data:
            raise ValueError("No results available. Run testing first.")
            
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Performance across scenarios
        plt.figure(figsize=(12, 6))
        scenarios = ['summer', 'winter', 'weekday', 'weekend']
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, (algo, results) in enumerate(plot_data.items()):
            means = [results[s]['mean_reward'] for s in scenarios]
            stds = [results[s]['std_reward'] for s in scenarios]
            plt.bar(x + i*width, means, width, label=algo.upper(),
                   yerr=stds, capsize=5)
            
        plt.xlabel('Scenario')
        plt.ylabel('Mean Reward')
        plt.title('Algorithm Performance Across Scenarios')
        plt.xticks(x + width, scenarios)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'scenario_comparison_{timestamp}.png'), dpi=300)
        plt.close()
        
        # 2. Energy consumption comparison
        plt.figure(figsize=(12, 6))
        for algo, results in plot_data.items():
            energy_data = [results[s]['mean_energy'] for s in scenarios]
            plt.plot(scenarios, energy_data, marker='o', label=algo.upper())
            
        plt.xlabel('Scenario')
        plt.ylabel('Mean Energy Consumption (kWh)')
        plt.title('Energy Consumption Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'energy_comparison_{timestamp}.png'), dpi=300)
        plt.close()
        
        # 3. Comfort violations
        plt.figure(figsize=(12, 6))
        for algo, results in plot_data.items():
            violations_data = [results[s]['mean_violations'] for s in scenarios]
            plt.plot(scenarios, violations_data, marker='o', label=algo.upper())
            
        plt.xlabel('Scenario')
        plt.ylabel('Mean Comfort Violations')
        plt.title('Comfort Violations Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'comfort_comparison_{timestamp}.png'), dpi=300)
        plt.close()
        
        # Save numerical results
        json_results = {
            'raw_results': self._convert_to_native_types(plot_data),
            'statistical_analysis': self._convert_to_native_types(self.statistical_analysis())
        }
        
        results_file = os.path.join(save_dir, f'testing_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=4)
            
        print(f"\nResults saved to {save_dir}")
        print(f"Results file: {results_file}")
        
    def clear_results(self):
        """Clear stored results."""
        self.results = {}
        self.current_results = {}


def main():
    """Main function to run model testing."""
    print("Starting model testing and validation...")
    
    # Create tester
    tester = ModelTester()
    
    # Compare algorithms
    print("\nComparing algorithms across scenarios...")
    results = tester.compare_algorithms(n_episodes=10)
    
    # Generate plots and analysis
    print("\nGenerating visualizations and analysis...")
    tester.plot_results()
    
    # Print statistical analysis
    analysis = tester.statistical_analysis()
    print("\nStatistical Analysis:")
    for comparison, stats in analysis.items():
        print(f"\n{comparison}:")
        print(f"  t-statistic: {stats['t_statistic']:.3f}")
        print(f"  p-value: {stats['p_value']:.3f}")
        print(f"  Significant difference: {stats['significant']}")
    
    print("\nTesting completed successfully!")


if __name__ == '__main__':
    main() 