import numpy as np
from rl_environment import SmartBuildingEnv
import pandas as pd
from data_manager import DataManager


def test_environment():
    """Test basic functionality of the SmartBuildingEnv."""
    
    try:
        # First load data to check columns
        print("\nLoading dataset to check columns...")
        data_manager = DataManager()
        df = data_manager.load_dataset('citylearn_dataset')
        print("\nAvailable columns:")
        for col in sorted(df.columns):
            print(f"  - {col}")
        print("\nIndex type:", type(df.index))
        print("First few timestamps:", df.index[:5])
        
        # Initialize environment
        print("\nInitializing environment...")
        env = SmartBuildingEnv(
            data_dir='citylearn_dataset',
            building_ids=[1, 2],  # Test with 2 buildings for simplicity
            time_step_minutes=15,
            episode_hours=24
        )
        
        # Test reset
        print("\nTesting environment reset...")
        obs, info = env.reset(seed=42)
        print("\nInitial observation structure:")
        print("\nBuildings:")
        for b_id, building_data in obs['buildings'].items():
            print(f"\n{b_id}:")
            for key, value in building_data.items():
                print(f"  {key}: shape={value.shape}, range=[{value.min():.2f}, {value.max():.2f}]")
        
        print("\nWeather:")
        for key, value in obs['weather'].items():
            print(f"  {key}: shape={value.shape}, range=[{value.min():.2f}, {value.max():.2f}]")
            
        print("\nCarbon Intensity:", obs['carbon_intensity'])
        print("Time Features:", obs['time_features'])
        print("\nInfo:", info)
        
        # Verify observation space
        print("\nVerifying observation space...")
        
        # Check buildings
        for b_id, building_data in obs['buildings'].items():
            print(f"\nChecking {b_id}:")
            building_space = env.observation_space.spaces['buildings'].spaces[b_id]
            for key, value in building_data.items():
                print(f"  {key}: ", end='')
                assert building_space.spaces[key].contains(value), f"Invalid {key} value"
                print("OK")
        
        # Check weather
        print("\nChecking weather:")
        weather_space = env.observation_space.spaces['weather']
        for key, value in obs['weather'].items():
            print(f"  {key}: ", end='')
            assert weather_space.spaces[key].contains(value), f"Invalid {key} value"
            print("OK")
        
        # Check other components
        print("\nChecking other components:")
        assert env.observation_space.spaces['carbon_intensity'].contains(obs['carbon_intensity']), "Invalid carbon_intensity"
        print("  carbon_intensity: OK")
        assert env.observation_space.spaces['time_features'].contains(obs['time_features']), "Invalid time_features"
        print("  time_features: OK")
        
        # Run one episode
        print("\nRunning test episode...")
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Random actions for each building
            actions = {
                b_id: env.action_space.spaces[b_id].sample()
                for b_id in env.action_space.spaces
            }
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Verify step outputs
            assert isinstance(reward, float), "Reward should be a float"
            assert isinstance(done, bool), "Done should be a boolean"
            
            # Print progress every 24 steps
            if steps % 24 == 0:
                print(f"Step {steps}/{env.steps_per_episode}, Reward: {reward:.2f}")
                # Print some building stats
                print("Building stats:")
                for b_id, b_info in info['buildings'].items():
                    print(f"  {b_id}:")
                    print(f"    Energy consumption: {b_info['energy_consumption']:.2f} kWh")
                    print(f"    Comfort penalty: {b_info['comfort_penalty']:.2f}")
                    print(f"    Target temperature: {b_info['target_temp']:.1f}°C")
        
        print(f"\nEpisode complete:")
        print(f"Steps taken: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Expected steps: {env.steps_per_episode}")
        assert steps == env.steps_per_episode, "Episode length mismatch"
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        
        if isinstance(e, KeyError):
            print("\nDataset column name mismatch. Available columns:")
            if hasattr(env, 'df'):
                print(env.df.columns.tolist())
        raise


if __name__ == '__main__':
    test_environment() 