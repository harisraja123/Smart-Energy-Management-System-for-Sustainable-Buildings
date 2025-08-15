import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import pandas as pd
from data_manager import DataManager


class SmartBuildingEnv(gym.Env):
    """Custom RL environment for smart building energy management."""
    
    def __init__(self, 
                 data_dir: str,
                 building_ids: Optional[List[str]] = None,
                 time_step_minutes: int = 15,
                 episode_hours: int = 24):
        """
        Initialize the environment.
        
        Args:
            data_dir: Directory containing building CSVs and weather data
            building_ids: List of building IDs to include, or None for all
            time_step_minutes: Time resolution in minutes (default: 15)
            episode_hours: Episode duration in hours (default: 24)
        """
        super().__init__()
        self.data_dir = data_dir
        # Load and preprocess data
        self.data_manager = DataManager()
        self.df = self.data_manager.load_dataset(data_dir)
        
        # Set building IDs
        all_buildings = list(range(1, 10))  # Buildings 1-9
        self.building_ids = building_ids if building_ids else all_buildings
        
        # Create datetime index from hour columns
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # Use Building_1 hour and month columns to create timestamps
            # Assume data starts at beginning of year
            year = 2020  # Example year
            hours = self.df['Building_1_hour'].values
            months = self.df['Building_1_month'].values
            days = np.ones_like(months)  # Start with day 1 for each month
            
            # Create timestamps
            timestamps = pd.to_datetime({
                'year': year,
                'month': months,
                'day': days,
                'hour': hours
            })
            
            # Set as index and handle duplicates by taking mean
            self.df = self.df.groupby(timestamps).mean()
            
        # Ensure index is sorted
        self.df = self.df.sort_index()
        
        # Create evenly spaced timestamps
        self.timestamps = pd.date_range(
            start=self.df.index.min(),
            end=self.df.index.max(),
            freq=f'{time_step_minutes}min'
        )
        
        # Resample data to match timestamps
        self.df = self.df.resample(f'{time_step_minutes}min').ffill()
        
        # Environment parameters
        self.time_step_minutes = time_step_minutes
        self.steps_per_episode = (episode_hours * 60) // time_step_minutes
        self.current_step = 0
        self.episode_start_idx = 0
        
        # Define state space components for each building
        # Continuous variables normalized to [0,1] range
        building_spaces = {}
        for b_id in self.building_ids:
            building_spaces[f'building_{b_id}'] = spaces.Dict({
                'temperature': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'humidity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'occupancy': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'non_shiftable_load': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'solar_generation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'cooling_demand': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'heating_demand': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'dhw_demand': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
        
        # Combined observation space
        self.observation_space = spaces.Dict({
            'buildings': spaces.Dict(building_spaces),
            'weather': spaces.Dict({
                'temperature': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'humidity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'solar_irradiance': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # diffuse, direct
            }),
            'carbon_intensity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'time_features': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # hour, day_of_week
        })
        
        # Define action space for each building
        # Temperature setpoints: 18-26°C in 0.5°C increments
        self.temp_setpoints = np.arange(18, 26.5, 0.5)
        self.action_space = spaces.Dict({
            f'building_{b_id}': spaces.Discrete(len(self.temp_setpoints))
            for b_id in self.building_ids
        })
        
        # Initialize thermal model parameters for each building
        self.thermal_params = {
            b_id: {
                'thermal_mass': 2000.0,  # kJ/K
                'insulation_coefficient': 0.5  # W/m²K
            }
            for b_id in self.building_ids
        }
        
        # Stochastic occupancy model parameters
        self.occupancy_patterns = {
            'office': {
                'weekday': {'start': 8, 'end': 18, 'base_prob': 0.8},
                'weekend': {'start': 10, 'end': 16, 'base_prob': 0.2}
            },
            'residential': {
                'weekday': {'start': 18, 'end': 8, 'base_prob': 0.9},
                'weekend': {'start': 9, 'end': 23, 'base_prob': 0.7}
            }
        }
        
        # Assign building types (example assignment)
        self.building_types = {
            1: 'office',    # Buildings 1-4 as office
            2: 'office',
            3: 'office',
            4: 'office',
            5: 'residential',  # Buildings 5-9 as residential
            6: 'residential',
            7: 'residential',
            8: 'residential',
            9: 'residential'
        }
        
        # Initialize random number generator
        self.np_random = np.random.RandomState()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to start new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        # Initialize random number generator
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Compute valid start positions at midnight that allow full episode length
        all_midnights = np.where(self.df.index.hour == 0)[0]
        valid_midnights = all_midnights[all_midnights <= len(self.df.index) - self.steps_per_episode]
        if len(valid_midnights) == 0:
            raise ValueError("No valid start times with full episode length in dataset")
        # Randomly select episode start position
        self.episode_start_idx = int(self.np_random.choice(valid_midnights))
        current_time = self.df.index[self.episode_start_idx]
        
        # Get initial state
        observation = self._get_state(current_time)
        info = {'timestamp': current_time}
        
        return observation, info
        
    def step(self, action: Union[Dict[str, int], int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Dictionary mapping building IDs to temperature setpoint indices
            
        Returns:
            observation: New state
            reward: Reward value
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Support single int actions by mapping to all buildings
        if isinstance(action, int):
            idx = action % len(self.temp_setpoints)
            action = {f'building_{b_id}': idx for b_id in self.building_ids}
        # Get current timestamp
        current_time = self.df.index[self.episode_start_idx + self.current_step]
        
        # Get current state
        current_state = self._get_state(current_time)
        
        # Calculate rewards for each building
        total_energy_cost = 0
        total_comfort_penalty = 0
        building_info = {}
        
        for b_id in self.building_ids:
            # Get building-specific action
            building_key = f'building_{b_id}'
            target_temp = self.temp_setpoints[action[building_key]]
            
            # Get building state
            building_state = current_state['buildings'][building_key]
            
            # Simulate thermal dynamics
            energy_consumption = self._simulate_thermal_dynamics(
                current_temp=building_state['temperature'][0] * 60 - 10,  # Denormalize
                target_temp=target_temp,
                outside_temp=current_state['weather']['temperature'][0] * 60 - 10,  # Denormalize
                building_id=b_id
            )
            
            # Calculate building-specific rewards
            energy_cost = (
                energy_consumption * 
                building_state['non_shiftable_load'][0] * 
                current_state['carbon_intensity'][0]  # Include carbon intensity in cost
            )
            
            comfort_penalty = self._calculate_comfort_penalty(
                current_temp=building_state['temperature'][0] * 60 - 10,
                target_temp=target_temp,
                occupancy=building_state['occupancy'][0]
            )
            
            # Accumulate total rewards
            total_energy_cost += energy_cost
            total_comfort_penalty += comfort_penalty
            
            # Store building-specific info
            building_info[building_key] = {
                'energy_consumption': energy_consumption,
                'comfort_penalty': comfort_penalty,
                'target_temp': target_temp
            }
        
        # Combined reward with carbon intensity weighting
        raw_reward = -0.7 * total_energy_cost - 0.3 * total_comfort_penalty
        
        # Scale reward but ensure meaningful differences between actions
        # Use a sigmoid-like transformation to bound rewards while preserving differences
        reward = -2.0 * (2.0 / (1.0 + np.exp(-raw_reward / 25.0)) - 1.0)
        
        # Advance simulation
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        
        # Get next state
        if not done:
            next_time = self.df.index[self.episode_start_idx + self.current_step]
            observation = self._get_state(next_time)
            info = {'timestamp': next_time, 'buildings': building_info}
        else:
            observation = current_state
            info = {'timestamp': current_time, 'buildings': building_info}
            
        return observation, reward, done, False, info
    
    def _get_state(self, timestamp: pd.Timestamp) -> Dict:
        """
        Construct state dictionary for given timestamp.
        
        Args:
            timestamp: Current simulation timestamp
            
        Returns:
            Dictionary containing all state components
        """
        row = self.df.loc[timestamp]
        
        # Build state dictionary
        state = {
            'buildings': {},
            'weather': {
                'temperature': np.array([row['weather_outdoor_dry_bulb_temperature']], dtype=np.float32),
                'humidity': np.array([row['weather_outdoor_relative_humidity'] / 100], dtype=np.float32),
                'solar_irradiance': np.array([
                    row['weather_diffuse_solar_irradiance'] / 1000,
                    row['weather_direct_solar_irradiance'] / 1000
                ], dtype=np.float32),
            },
            'carbon_intensity': np.array([row['carbon_intensity_carbon_intensity'] / 100], dtype=np.float32),
            'time_features': np.array([
                timestamp.hour / 24,
                timestamp.dayofweek / 6
            ], dtype=np.float32)
        }
        
        # Add state for each building
        for b_id in self.building_ids:
            prefix = f'Building_{b_id}'
            state['buildings'][f'building_{b_id}'] = {
                'temperature': np.array([row[f'{prefix}_indoor_dry_bulb_temperature']], dtype=np.float32),
                'humidity': np.array([row[f'{prefix}_indoor_relative_humidity'] / 100], dtype=np.float32),
                'occupancy': np.array([self._get_stochastic_occupancy(timestamp, b_id)], dtype=np.float32),
                'non_shiftable_load': np.array([row[f'{prefix}_non_shiftable_load'] / 100], dtype=np.float32),
                'solar_generation': np.array([row.get(f'{prefix}_solar_generation', 0) / 100], dtype=np.float32),
                'cooling_demand': np.array([row[f'{prefix}_cooling_demand'] / 100], dtype=np.float32),
                'heating_demand': np.array([row[f'{prefix}_heating_demand'] / 100], dtype=np.float32),
                'dhw_demand': np.array([row.get(f'{prefix}_dhw_demand', 0) / 100], dtype=np.float32),
            }
        
        return state
    
    def _simulate_thermal_dynamics(self, 
                                 current_temp: float,
                                 target_temp: float,
                                 outside_temp: float,
                                 building_id: int) -> float:
        """
        Simulate building thermal dynamics using a simple thermal model.
        
        Args:
            current_temp: Current indoor temperature (°C)
            target_temp: Target temperature setpoint (°C)
            outside_temp: Outside temperature (°C)
            building_id: ID of the building being simulated
            
        Returns:
            Energy consumption for this timestep (kWh)
        """
        params = self.thermal_params[building_id]
        
        # Temperature difference driving heat transfer
        delta_t_outside = outside_temp - current_temp
        delta_t_target = target_temp - current_temp
        
        # Heat transfer through building envelope
        q_envelope = params['insulation_coefficient'] * delta_t_outside
        
        # Required heating/cooling energy
        q_hvac = params['thermal_mass'] * delta_t_target / (self.time_step_minutes * 60)
        
        # Total energy consumption (convert to kWh)
        energy = abs(q_hvac + q_envelope) * self.time_step_minutes / 60 / 1000
        
        return energy
    
    def _calculate_comfort_penalty(self,
                                 current_temp: float,
                                 target_temp: float,
                                 occupancy: float) -> float:
        """
        Calculate comfort penalty based on temperature deviation and occupancy.
        
        Args:
            current_temp: Current indoor temperature (°C)
            target_temp: Target temperature setpoint (°C)
            occupancy: Current occupancy level [0,1]
            
        Returns:
            Comfort penalty value [0,1]
        """
        # Only penalize comfort when space is occupied
        if occupancy < 0.1:
            return 0.0
            
        # Calculate temperature deviation
        temp_dev = abs(current_temp - target_temp)
        
        # Penalty increases quadratically with deviation
        penalty = min((temp_dev / 2) ** 2, 1.0)
        
        return penalty
    
    def _get_stochastic_occupancy(self, timestamp: pd.Timestamp, building_id: int) -> float:
        """
        Generate stochastic occupancy based on time patterns.
        
        Args:
            timestamp: Current simulation time
            building_id: ID of the building
            
        Returns:
            Occupancy level [0,1]
        """
        # Get building type
        building_type = self.building_types[building_id]
        
        # Get pattern based on weekday/weekend
        is_weekend = timestamp.dayofweek >= 5
        pattern = self.occupancy_patterns[building_type]['weekend' if is_weekend else 'weekday']
        
        # Check if within typical occupied hours
        hour = timestamp.hour
        in_hours = pattern['start'] <= hour <= pattern['end']
        
        # Base probability from pattern
        base_prob = pattern['base_prob'] if in_hours else 0.1
        
        # Add random variation
        variation = self.np_random.normal(0, 0.1)
        prob = np.clip(base_prob + variation, 0, 1)
        
        return float(prob) 