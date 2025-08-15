import warnings
warnings.filterwarnings('ignore', message='Couldn\'t import dot_parser.*')

import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    if args and isinstance(args[0], str) and "Couldn't import dot_parser" in args[0]:
        return
    _orig_print(*args, **kwargs)
builtins.print = print

import sys
_orig_stderr_write = sys.stderr.write
def _stderr_write_filter(msg):
    if "Producer process has been terminated before all shared CUDA tensors released" in msg:
        return
    return _orig_stderr_write(msg)
sys.stderr.write = _stderr_write_filter

# Import required modules with error handling
try:
    import argparse
    import os
    import json
    import torch
    import numpy as np
    from datetime import datetime
    from typing import Dict, List
    import matplotlib.pyplot as plt
    import psutil
    import subprocess
    import time
    import webbrowser

    from data_manager import DataManager
    from rl_environment import SmartBuildingEnv
    from rl_agents import DQNAgent, PPOAgent, A3CAgent
    from test_models import ModelTester
    from dashboard.utils import TrainingMonitor
except ImportError as e:
    print(f"\nError importing required modules: {str(e)}")
    print("Please ensure all dependencies are installed by running:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def validate_data_dir(data_dir: str) -> bool:
    """
    Validate that the data directory contains required files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        bool: True if validation passes
    """
    try:
        # Check directory exists
        if not os.path.isdir(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            return False
            
        # Check for required files
        required_patterns = [
            'Building_*.csv',  # Building data files
            'weather.csv',     # Weather data
            'carbon_intensity.csv'  # Carbon intensity data
        ]
        
        import glob
        for pattern in required_patterns:
            files = glob.glob(os.path.join(data_dir, pattern))
            if not files:
                print(f"Error: No files matching {pattern} found in {data_dir}")
                return False
                
        # Check building files if specific buildings requested
        building_files = glob.glob(os.path.join(data_dir, 'Building_*.csv'))
        available_buildings = sorted([
            int(os.path.basename(f).split('_')[1].split('.')[0])
            for f in building_files
        ])
        
        return True
        
    except Exception as e:
        print(f"Error validating data directory: {str(e)}")
        return False


def train_dqn(env: SmartBuildingEnv,
              n_episodes: int = 1000,
              eval_interval: int = 10,
              save_dir: str = 'models',
              patience: int = 5000,
              min_reward_threshold: float = 10.0,
              min_episodes: int = 100) -> Dict:
    """Train DQN agent."""
    print("Initializing DQN training...")
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    monitor.start_training('DQN', n_episodes)
    
    try:
        # Create agent
        state_dims = {
            'building': 8 * len(env.building_ids)  # 8 features per building
        }
        
        # Get action dimensions for each building
        action_dims = {
            f'building_{building_id}': env.action_space[f'building_{building_id}'].n
            for building_id in env.building_ids
        }
        
        print("Creating DQN agent with parameters:")
        print(f"State dimensions: {state_dims}")
        print(f"Action dimensions: {action_dims}")
        print(f"Early stopping patience: {patience}")
        print(f"Minimum reward threshold: {min_reward_threshold}")
        print(f"Minimum episodes: {min_episodes}")
        
        agent = DQNAgent(
            state_dims=state_dims,
            action_dims=action_dims,
            learning_rate=3e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.99,
            buffer_size=100000,
            batch_size=1024,
            target_update=10,
            n_envs=1,
            patience=patience,
            min_reward_threshold=min_reward_threshold,
            min_episodes=min_episodes
        )
        
        print("DQN agent created successfully")
        print("Starting training loop...")
        
        # Training loop
        rewards_history = []
        best_reward = float('-inf')
        
        for episode in range(n_episodes):
            print(f"\nStarting episode {episode + 1}")
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            total_energy = 0
            
            while not done:
                # Select and take action
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                # Update agent
                loss = agent.update()
                
                episode_reward += reward
                # Track energy if available in info
                if info and 'energy' in info:
                    total_energy += info['energy']
                state = next_state
                steps += 1
            
            # Calculate mean energy for this episode
            mean_energy = total_energy / steps if steps > 0 else 0
            
            # Update monitor with supported parameters
            monitor.update_progress(
                episode=episode + 1,
                reward=episode_reward,
                total_episodes=n_episodes,
                mean_energy=mean_energy,
                mean_violations=0,  # Add if available from environment
                std_reward=0,  # Calculate if needed
                is_training=True
            )
            
            # Print additional metrics not shown in monitor
            print(f"Epsilon: {agent.epsilon:.3f}")
            if loss is not None:
                print(f"Loss: {loss:.3f}")
            
            rewards_history.append(episode_reward)
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                print("\nRunning evaluation...")
                eval_reward = evaluate_agent(env, agent)
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Training reward: {episode_reward:.2f}")
                print(f"Evaluation reward: {eval_reward:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                if loss is not None:
                    print(f"Loss: {loss:.3f}")
                print()
                
                # Early stopping check
                agent.total_episodes = episode + 1
                
                if eval_reward > agent.best_reward:
                    agent.best_reward = eval_reward
                    agent.episodes_without_improvement = 0
                    agent.last_best_update = episode
                    save_model(agent, 'dqn', save_dir, is_best=True)
                    print(f"New best model saved with reward: {agent.best_reward:.2f}")
                else:
                    agent.episodes_without_improvement += eval_interval
                    episodes_since_improvement = episode - agent.last_best_update
                    print(f"Episodes without improvement: {episodes_since_improvement}")
                    
                    # Early stopping checks
                    if agent.total_episodes >= agent.min_episodes:
                        if agent.best_reward >= agent.min_reward_threshold:
                            print(f"\nStopping training - Reached minimum reward threshold: {agent.best_reward:.2f}")
                            # Update progress to 100% for early stopping
                            monitor.update_progress(
                                episode=n_episodes,
                                reward=agent.best_reward,
                                total_episodes=n_episodes,
                                mean_energy=mean_energy,
                                mean_violations=0,
                                std_reward=0,
                                is_training=True
                            )
                            break
                        elif episodes_since_improvement >= agent.patience:
                            print(f"\nStopping training - No improvement for {episodes_since_improvement} episodes")
                            # Update progress to 100% for early stopping
                            monitor.update_progress(
                                episode=n_episodes,
                                reward=agent.best_reward,
                                total_episodes=n_episodes,
                                mean_energy=mean_energy,
                                mean_violations=0,
                                std_reward=0,
                                is_training=True
                            )
                            break
                    
        return {'rewards': rewards_history, 'best_reward': agent.best_reward}
        
    finally:
        # End training monitoring
        monitor.end_training()


def train_ppo(env: SmartBuildingEnv,
              n_episodes: int = 1000,
              eval_interval: int = 10,
              save_dir: str = 'models',
              patience: int = 5000,
              min_reward_threshold: float = 10.0,
              min_episodes: int = 100) -> Dict:
    """Train PPO agent."""
    # Create agent
    state_dims = {'building': 8 * len(env.building_ids)}
    n_actions = len(env.temp_setpoints) * len(env.building_ids)
    
    print("Creating PPO agent with parameters:")
    print(f"State dimensions: {state_dims}")
    print(f"Number of actions: {n_actions}")
    print(f"Early stopping patience: {patience}")
    print(f"Minimum reward threshold: {min_reward_threshold}")
    print(f"Minimum episodes: {min_episodes}")
    
    agent = PPOAgent(
        state_dims=state_dims,
        n_actions=n_actions,
        learning_rate=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        patience=patience,
        min_reward_threshold=min_reward_threshold,
        min_episodes=min_episodes
    )
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    monitor.start_training('PPO', n_episodes)
    
    try:
        # Training loop
        rewards_history = []
        best_reward = float('-inf')
        
        for episode in range(n_episodes):
            # Collect episode experience
            states, actions, rewards = [], [], []
            values, policies = [], []
            
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Select and take action
                action, value, policy = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                policies.append(policy)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
            # Update agent
            next_states = states[1:] + [state]
            metrics = agent.update(states, actions, rewards, values, policies, next_states, [False] * (len(states)-1) + [done])
            
            rewards_history.append(episode_reward)
            
            # Update monitor
            monitor.update_progress(
                episode=episode + 1,
                reward=episode_reward,
                total_episodes=n_episodes,
                mean_energy=sum(rewards) / len(rewards) if rewards else 0,
                mean_violations=0,  # Add if available from environment
                std_reward=0,  # Calculate if needed
                is_training=True
            )
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_reward = evaluate_agent(env, agent)
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Training reward: {episode_reward:.2f}")
                print(f"Evaluation reward: {eval_reward:.2f}")
                print(f"Policy loss: {metrics['policy_loss']:.3f}")
                print(f"Value loss: {metrics['value_loss']:.3f}")
                print(f"Entropy: {metrics['entropy']:.3f}")
                print()
                
                # Early stopping check
                agent.total_episodes = episode + 1
                
                if eval_reward > agent.best_reward:
                    agent.best_reward = eval_reward
                    agent.episodes_without_improvement = 0
                    agent.last_best_update = episode
                    save_model(agent, 'ppo', save_dir, is_best=True)
                    print(f"New best model saved with reward: {agent.best_reward:.2f}")
                else:
                    agent.episodes_without_improvement += eval_interval
                    episodes_since_improvement = episode - agent.last_best_update
                    print(f"Episodes without improvement: {episodes_since_improvement}")
                    
                    # Early stopping checks
                    if agent.total_episodes >= agent.min_episodes:
                        if agent.best_reward >= agent.min_reward_threshold:
                            print(f"\nStopping training - Reached minimum reward threshold: {agent.best_reward:.2f}")
                            # Update progress to 100% for early stopping
                            monitor.update_progress(
                                episode=n_episodes,
                                reward=agent.best_reward,
                                total_episodes=n_episodes,
                                mean_energy=sum(rewards) / len(rewards) if rewards else 0,
                                mean_violations=0,
                                std_reward=0,
                                is_training=True
                            )
                            break
                        elif episodes_since_improvement >= agent.patience:
                            print(f"\nStopping training - No improvement for {episodes_since_improvement} episodes")
                            # Update progress to 100% for early stopping
                            monitor.update_progress(
                                episode=n_episodes,
                                reward=agent.best_reward,
                                total_episodes=n_episodes,
                                mean_energy=sum(rewards) / len(rewards) if rewards else 0,
                                mean_violations=0,
                                std_reward=0,
                                is_training=True
                            )
                            break
                    
        return {'rewards': rewards_history, 'best_reward': agent.best_reward}
        
    finally:
        # End training monitoring
        monitor.end_training()


def train_a3c(env: SmartBuildingEnv,
              n_episodes: int = 1000,
              n_workers: int = 4,
              eval_interval: int = 10,
              save_dir: str = 'models') -> Dict:
    """Train A3C agent."""
    # Create agent
    state_dims = {'building': 8 * len(env.building_ids)}
    n_actions = len(env.temp_setpoints) * len(env.building_ids)
    
    def env_creator():
        return SmartBuildingEnv(
            data_dir=env.data_dir,
            building_ids=env.building_ids,
            time_step_minutes=env.time_step_minutes,
            episode_hours=24
        )
    
    agent = A3CAgent(
        state_dims=state_dims,
        n_actions=n_actions,
        env_creator=env_creator,
        n_workers=n_workers,
        learning_rate=1e-4,
        gamma=0.99
    )
    
    # Initialize training monitor
    from dashboard.utils import TrainingMonitor
    monitor = TrainingMonitor()
    monitor.start_training('A3C', n_episodes)
    
    try:
        # Start training
        results = agent.train(eval_interval=eval_interval, save_dir=save_dir, max_episodes=n_episodes)
        
        # Update monitor with final results
        if results:
            # Check if training was stopped early
            actual_episodes = results.get('total_episodes', n_episodes)
            early_stopped = actual_episodes < n_episodes
            
            # Always update to 100% if training completed (either naturally or through early stopping)
            monitor.update_progress(
                episode=n_episodes,
                reward=results.get('best_reward', 0),
                total_episodes=n_episodes,
                mean_energy=results.get('mean_energy', 0),
                mean_violations=results.get('mean_violations', 0),
                std_reward=results.get('std_reward', 0),
                is_training=True
            )
            
            if early_stopped:
                print(f"\nTraining stopped early at episode {actual_episodes}")
    finally:
        # Stop training and update monitor
        agent.stop()
        monitor.end_training()
    
    return results


def evaluate_agent(env: SmartBuildingEnv, agent, n_episodes: int = 5) -> float:
    """Evaluate agent performance."""
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action (handles PPO returning a tuple)
            res = agent.select_action(state, training=False)
            action = res[0] if isinstance(res, tuple) else res
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        total_reward += episode_reward
        
    return total_reward / n_episodes


def plot_training_progress(rewards_history, algorithm, save_dir):
    """Plot training rewards and save figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title(f'{algorithm.upper()} Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Add moving average
    if len(rewards_history) > 10:
        window_size = min(10, len(rewards_history) // 5)
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, 'r-', linewidth=2)
        plt.legend(['Episode Reward', f'{window_size}-Episode Moving Avg'])
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(save_dir, f'{algorithm}_training_progress_{timestamp}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return plot_path

def save_model(agent, algorithm: str, save_dir: str, is_best: bool = False):
    """
    Save model weights and configuration.
    
    Args:
        agent: The RL agent to save
        algorithm: Algorithm name ('dqn', 'ppo', or 'a3c')
        save_dir: Directory to save the model
        is_best: Whether this is the best model so far
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model weights with timestamp
    model_path = os.path.join(save_dir, f'{algorithm}_model_{timestamp}.pt')
    if algorithm == 'dqn':
        torch.save(agent.policy_net.state_dict(), model_path)
    elif algorithm == 'ppo':
        torch.save(agent.network.state_dict(), model_path)
    else:  # a3c
        torch.save(agent.global_network.state_dict(), model_path)
        
    # Save configuration with timestamp
    config = {
        'algorithm': algorithm,
        'timestamp': timestamp,
        'model_path': model_path,
        'is_best': is_best
    }
    config_path = os.path.join(save_dir, f'{algorithm}_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    # If this is the best model, save an additional copy with 'best' in the name
    if is_best:
        best_model_path = os.path.join(save_dir, f'{algorithm}_model_best.pt')
        best_config_path = os.path.join(save_dir, f'{algorithm}_config_best.json')
        
        # Save best model weights
        if algorithm == 'dqn':
            torch.save(agent.policy_net.state_dict(), best_model_path)
        elif algorithm == 'ppo':
            torch.save(agent.network.state_dict(), best_model_path)
        else:  # a3c
            torch.save(agent.global_network.state_dict(), best_model_path)
            
        # Save best model config
        best_config = {
            'algorithm': algorithm,
            'timestamp': timestamp,
            'model_path': best_model_path,
            'is_best': True,
            'original_model_path': model_path
        }
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=4)


def train_with_dashboard(algorithm: str, env: SmartBuildingEnv, **kwargs):
    """Training function with Streamlit dashboard integration"""
    import subprocess
    import threading
    import time
    import webbrowser
    import os
    import psutil
    
    def is_streamlit_running():
        """Check if Streamlit is already running on port 8501"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' and proc.info['cmdline']:
                    if 'streamlit' in ' '.join(proc.info['cmdline']) and '8501' in ' '.join(proc.info['cmdline']):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    
    # Ensure dashboard directory exists
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
    if not os.path.exists(dashboard_dir):
        print(f"Creating dashboard directory at {dashboard_dir}")
        os.makedirs(dashboard_dir, exist_ok=True)
    
    # Get the absolute path to app.py
    dashboard_path = os.path.join(dashboard_dir, 'app.py')
    
    # Only start dashboard if it's not already running
    dashboard_process = None
    if not is_streamlit_running():
        print("\nLaunching Streamlit dashboard...")
        dashboard_process = subprocess.Popen(
            ["streamlit", "run", dashboard_path, "--server.port", "8501", "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait for dashboard to start
        time.sleep(3)
        
        # Open dashboard in browser only once
        webbrowser.open('http://localhost:8501')
        
        print("\nDashboard is running at http://localhost:8501")
        print("If the dashboard doesn't open automatically, please open the URL in your browser.")
    else:
        print("\nDashboard is already running at http://localhost:8501")
    
    print("\nStarting training...\n")
    results = None
    
    try:
        # Initialize training monitor
        from dashboard.utils import TrainingMonitor
        monitor = TrainingMonitor()
        
        # Clear any existing data and start new training session
        monitor.clear_all()
        monitor.start_training(algorithm, kwargs.get('n_episodes', 1000))
        
        # Run training based on algorithm
        if algorithm == 'dqn':
            results = train_dqn(env, **kwargs)
        elif algorithm == 'ppo':
            results = train_ppo(env, **kwargs)
        else:  # a3c
            results = train_a3c(env, **kwargs)
            
        # Save training progress plot if we have results
        if results and 'rewards' in results:
            plot_path = plot_training_progress(results['rewards'], algorithm, kwargs.get('save_dir', 'models'))
            print(f"\nTraining progress plot saved to {plot_path}")
            
            # Save results to JSON
            save_dir = kwargs.get('save_dir', 'models')
            results_path = os.path.join(save_dir, f'{algorithm}_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Training results saved to {results_path}")
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
        
    finally:
        if results:
            print("\nTraining completed!")
            print("\nDashboard is still running for result visualization.")
            print("You can continue to explore the results in the dashboard.")
            print("\nPress Enter to close the dashboard and exit...")
            input()
        
        # Only cleanup if we started the process
        if dashboard_process:
            print("\nShutting down dashboard...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()
    
    return results

def test_with_dashboard(tester: ModelTester, algorithm: str, n_episodes: int):
    """Run testing with dashboard visualization."""
    import subprocess
    import time
    import webbrowser
    import os
    import psutil
    from dashboard.utils import TrainingMonitor
    
    def is_streamlit_running():
        """Check if Streamlit is already running on port 8501"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' and proc.info['cmdline']:
                    if 'streamlit' in ' '.join(proc.info['cmdline']) and '8501' in ' '.join(proc.info['cmdline']):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    
    # Ensure dashboard directory exists
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
    dashboard_path = os.path.join(dashboard_dir, 'app.py')
    
    # Start dashboard if not running
    dashboard_process = None
    results = None
    monitor = TrainingMonitor()
    
    try:
        if not is_streamlit_running():
            print("\nLaunching Streamlit dashboard...")
            dashboard_process = subprocess.Popen(
                ["streamlit", "run", dashboard_path, "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
            print("\nDashboard is running at http://localhost:8501")
        else:
            print("\nDashboard is already running at http://localhost:8501")
        
        # Clear any previous results
        tester.clear_results()
        
        if algorithm == 'all':
            # Run comparison testing for all algorithms
            algorithms = ['dqn', 'ppo', 'a3c']
            scenarios = ['summer', 'winter', 'weekday', 'weekend']
            all_results = {}
            
            # Initialize monitor for comparison testing
            monitor.start_testing("ALL ALGORITHMS", n_episodes)
            
            for algo in algorithms:
                # Check if model exists before testing
                if algo not in tester.best_models:
                    print(f"\nNo trained model found for {algo.upper()}. Skipping...")
                    continue
                    
                algo_results = {}
                for i, scenario in enumerate(scenarios):
                    try:
                        print(f"\nTesting {algo.upper()} in {scenario} scenario...")
                        results = tester.run_scenario_test(algo, scenario, n_episodes)
                        
                        if results:
                            algo_results[scenario] = results
                            # Update dashboard with results
                            monitor.update_progress(
                                episode=i + 1,
                                reward=results['mean_reward'],
                                total_episodes=len(scenarios),
                                scenario=f"{algo.upper()} - {scenario}",
                                mean_energy=results['mean_energy'],
                                mean_violations=results['mean_violations'],
                                std_reward=results['std_reward'],
                                is_training=False
                            )
                    except Exception as e:
                        print(f"\nError testing {algo.upper()} in {scenario} scenario: {str(e)}")
                        continue
                
                if algo_results:
                    all_results[algo] = algo_results
                    # Save individual algorithm results
                    try:
                        # Calculate aggregate metrics across scenarios
                        mean_reward = np.mean([r['mean_reward'] for r in algo_results.values()])
                        std_reward = np.mean([r['std_reward'] for r in algo_results.values()])
                        mean_energy = np.mean([r['mean_energy'] for r in algo_results.values()])
                        mean_violations = np.mean([r['mean_violations'] for r in algo_results.values()])
                        
                        algo_summary = {
                            'mean_reward': float(mean_reward),
                            'std_reward': float(std_reward),
                            'mean_energy': float(mean_energy),
                            'mean_violations': float(mean_violations),
                            'scenarios': algo_results
                        }
                        
                        # Save individual algorithm results
                        algo_results_path = os.path.join(tester.models_dir, f'{algo}_test_results.json')
                        with open(algo_results_path, 'w') as f:
                            json.dump(algo_summary, f, indent=4)
                        print(f"\nSaved {algo.upper()} results to {algo_results_path}")
                    except Exception as e:
                        print(f"\nError saving {algo.upper()} results: {str(e)}")
            
            results = all_results
            
            # Save combined results
            try:
                results_path = os.path.join(tester.models_dir, 'all_test_results.json')
                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=4)
                print(f"\nSaved combined results to {results_path}")
                
                # Generate comparison plots
                print("\nGenerating comparison plots...")
                os.makedirs('results', exist_ok=True)
                tester.plot_results(save_dir='results')
                
            except Exception as e:
                print(f"\nError saving combined results: {str(e)}")
            
        else:
            # Testing single algorithm
            monitor.start_testing(f"{algorithm.upper()}", n_episodes)
            print(f"\nTesting {algorithm.upper()} across scenarios...")
            scenarios = ['summer', 'winter', 'weekday', 'weekend']
            
            # Check if model exists before testing
            if algorithm not in tester.best_models:
                print(f"\nNo trained model found for {algorithm.upper()}. Please train the model first.")
                return None
                
            algo_results = {}
            for i, scenario in enumerate(scenarios):
                try:
                    print(f"\nTesting in {scenario} scenario...")
                    results = tester.run_scenario_test(algorithm, scenario, n_episodes)
                    
                    if results:
                        algo_results[scenario] = results
                        # Update dashboard with results
                        monitor.update_progress(
                            episode=i + 1,
                            reward=results['mean_reward'],
                            total_episodes=len(scenarios),
                            scenario=scenario,
                            mean_energy=results['mean_energy'],
                            mean_violations=results['mean_violations'],
                            std_reward=results['std_reward'],
                            is_training=False
                        )
                except Exception as e:
                    print(f"\nError testing {scenario} scenario: {str(e)}")
                    continue
            
            # Save results for single algorithm
            if algo_results:
                try:
                    # Calculate aggregate metrics across scenarios
                    mean_reward = np.mean([r['mean_reward'] for r in algo_results.values()])
                    std_reward = np.mean([r['std_reward'] for r in algo_results.values()])
                    mean_energy = np.mean([r['mean_energy'] for r in algo_results.values()])
                    mean_violations = np.mean([r['mean_violations'] for r in algo_results.values()])
                    
                    results = {
                        'mean_reward': float(mean_reward),
                        'std_reward': float(std_reward),
                        'mean_energy': float(mean_energy),
                        'mean_violations': float(mean_violations),
                        'scenarios': algo_results
                    }
                    
                    # Save results
                    results_path = os.path.join(tester.models_dir, f'{algorithm}_test_results.json')
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    print(f"\nSaved results to {results_path}")
                    
                    # Generate plots for single algorithm
                    print("\nGenerating plots...")
                    os.makedirs('results', exist_ok=True)
                    tester.plot_results(save_dir='results')
                    
                except Exception as e:
                    print(f"\nError saving results: {str(e)}")
        
        if results:
            print("\nTesting completed! Results are available in the dashboard.")
            print("Press Enter to close the dashboard and exit...")
            input()
        else:
            print("\nNo test results were generated. Please check for errors above.")
        
        return results
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return None
        
    finally:
        # End testing monitoring
        monitor.end_testing()
        
        if dashboard_process:
            print("\nShutting down dashboard...")
            try:
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"Error shutting down dashboard gracefully: {str(e)}")
                try:
                    dashboard_process.kill()
                except Exception:
                    pass  # Ignore any errors in force kill

def main():
    """Main entry point."""
    import argparse
    import sys
    
    try:
        parser = argparse.ArgumentParser(description='Smart Building Energy Management with RL')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                          help='Mode of operation: train or test')
        parser.add_argument('--algorithm', type=str, default='a3c', choices=['dqn', 'ppo', 'a3c', 'all'],
                          help='RL algorithm to use')
        parser.add_argument('--data_dir', type=str, default='citylearn_dataset',
                          help='Directory containing the dataset')
        parser.add_argument('--buildings', type=int, nargs='+',
                          help='List of building IDs to use')
        parser.add_argument('--n_episodes', type=int, default=1000,
                          help='Number of episodes for training')
        parser.add_argument('--test_episodes', type=int, default=10,
                          help='Number of episodes for testing')
        parser.add_argument('--eval_interval', type=int, default=10,
                          help='Interval for evaluation during training')
        parser.add_argument('--save_dir', type=str, default='models',
                          help='Directory to save models and results')
        parser.add_argument('--with_dashboard', action='store_true',
                          help='Enable Streamlit dashboard')
                          
        args = parser.parse_args()
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("Using device: cuda")
            print("GPU Available")
        else:
            print("Using device: cpu")
            print("No GPU available")
        
        # Validate data directory first
        if not validate_data_dir(args.data_dir):
            print("\nPlease check the data directory and try again.")
            sys.exit(1)
        
        # Load data and check available buildings
        try:
            data_manager = DataManager()
            available_buildings = data_manager.get_available_buildings(args.data_dir)
            print(f"Found data for buildings: {available_buildings}")
            
            # Use all available buildings if none specified
            if not args.buildings:
                args.buildings = available_buildings
            else:
                # Validate requested buildings exist
                invalid_buildings = [b for b in args.buildings if b not in available_buildings]
                if invalid_buildings:
                    print(f"\nError: Requested buildings {invalid_buildings} not found in dataset.")
                    print(f"Available buildings: {available_buildings}")
                    sys.exit(1)
                
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            print("Please check that the data directory exists and contains valid data files.")
            sys.exit(1)
        
        try:
            if args.mode == 'train':
                # Create environment with error handling
                try:
                    env = SmartBuildingEnv(
                        data_dir=args.data_dir,
                        building_ids=args.buildings,
                        time_step_minutes=15,
                        episode_hours=24
                    )
                except Exception as e:
                    print(f"\nError creating environment: {str(e)}")
                    print("Please check environment configuration and data files.")
                    sys.exit(1)
                
                # Use dashboard if requested
                if args.with_dashboard:
                    results = train_with_dashboard(
                        args.algorithm,
                        env,
                        n_episodes=args.n_episodes,
                        eval_interval=args.eval_interval,
                        save_dir=args.save_dir
                    )
                else:
                    if args.algorithm == 'dqn':
                        results = train_dqn(env, args.n_episodes, args.eval_interval, args.save_dir)
                    elif args.algorithm == 'ppo':
                        results = train_ppo(env, args.n_episodes, args.eval_interval, args.save_dir)
                    else:  # a3c
                        results = train_a3c(env, args.n_episodes, 4, args.eval_interval, args.save_dir)
                    
                    if results:
                        try:
                            # Generate and save training progress plot
                            plot_path = plot_training_progress(results['rewards'], args.algorithm, args.save_dir)
                            
                            # Save results
                            results_path = os.path.join(args.save_dir, f'{args.algorithm}_results.json')
                            with open(results_path, 'w') as f:
                                json.dump(results, f, indent=4)
                                
                            print(f"\nTraining completed. Results saved to {results_path}")
                            print(f"Training progress plot saved to {plot_path}")
                        except Exception as e:
                            print(f"\nError saving results: {str(e)}")
                            print("Training completed but results could not be saved.")
                    else:
                        print("\nTraining did not complete successfully. Please check for errors above.")
                        
            else:  # Test mode
                print("\nStarting model testing and validation...")
                
                try:
                    # Create model tester
                    tester = ModelTester(args.save_dir, args.data_dir)
                    
                    # Run testing with or without dashboard
                    if args.with_dashboard:
                        results = test_with_dashboard(tester, args.algorithm, args.test_episodes)
                    else:
                        if args.algorithm == 'all':
                            results = tester.run_comparison_test(args.test_episodes)
                            if results:
                                try:
                                    # Save test results
                                    results_path = os.path.join(args.save_dir, 'all_test_results.json')
                                    with open(results_path, 'w') as f:
                                        json.dump(results, f, indent=4)
                                    print(f"\nTest results saved to {results_path}")
                                    
                                    # Generate comparison plots
                                    print("\nGenerating comparison plots...")
                                    os.makedirs('results', exist_ok=True)
                                    tester.plot_results(save_dir='results')
                                    
                                except Exception as e:
                                    print(f"\nError saving test results: {str(e)}")
                        else:
                            results = tester.run_scenario_test(args.algorithm, 'summer', args.test_episodes)
                            if results:
                                try:
                                    # Save test results
                                    results_path = os.path.join(args.save_dir, f'{args.algorithm}_test_results.json')
                                    with open(results_path, 'w') as f:
                                        json.dump(results, f, indent=4)
                                    print(f"\nTest results saved to {results_path}")
                                    
                                    # Generate plots
                                    print("\nGenerating plots...")
                                    os.makedirs('results', exist_ok=True)
                                    tester.plot_results(save_dir='results')
                                    
                                except Exception as e:
                                    print(f"\nError saving test results: {str(e)}")
                                    
                    if results:
                        try:
                            # Save test results
                            results_path = os.path.join(args.save_dir, f'{args.algorithm}_test_results.json')
                            with open(results_path, 'w') as f:
                                json.dump(results, f, indent=4)
                            print(f"\nTest results saved to {results_path}")
                        except Exception as e:
                            print(f"\nError saving test results: {str(e)}")
                    else:
                        print("\nNo test results were generated. Please check for errors above.")
                        
                except Exception as e:
                    print(f"\nError during testing: {str(e)}")
                    print("Please ensure that trained models exist and the test configuration is correct.")
                    sys.exit(1)
                    
        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 