import os
import json
from datetime import datetime
import threading
from typing import Dict, Any
import numpy as np
import time
import portalocker  # Cross-platform file locking

class TrainingMonitor:
    """Monitor and update training/testing progress."""
    
    def __init__(self):
        """Initialize the monitor."""
        self.training_file = os.path.join('dashboard', 'data', 'current_training.json')
        self.testing_file = os.path.join('dashboard', 'data', 'current_testing.json')
        self.lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.training_file), exist_ok=True)
        
        # Initialize both files with default data
        self._init_file(self.training_file, is_training=True)
        self._init_file(self.testing_file, is_training=False)
    
    def _safe_write_json(self, filepath: str, data: Dict):
        """Safely write JSON data with file locking."""
        max_retries = 3
        retry_delay = 0.1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use portalocker for cross-platform file locking
                with portalocker.Lock(filepath, 'w', timeout=10) as f:
                    json.dump(data, f, indent=4)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force write to disk
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                print(f"Error writing to {filepath}: {str(e)}")
                return False
    
    def _safe_read_json(self, filepath: str) -> Dict:
        """Safely read JSON data with file locking."""
        max_retries = 3
        retry_delay = 0.1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use portalocker for cross-platform file locking
                with portalocker.Lock(filepath, 'r', timeout=10) as f:
                    data = json.load(f)
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise
    
    def _init_file(self, filepath: str, is_training: bool):
        """Initialize a JSON file with default data."""
        default_data = {
            'episode': 0,
            'total_episodes': 0,
            'reward': 0.0,
            'std_reward': 0.0,
            'mean_energy': 0.0,
            'mean_violations': 0.0,
            'scenario': 'N/A',
            'is_training': is_training,
            'is_testing': not is_training,
            'algorithm': 'N/A',
            'active': False,
            f'{"training" if is_training else "testing"}_start_time': None,
            'scenarios': {},
            'completed': False
        }
        self._safe_write_json(filepath, default_data)
    
    def start_training(self, algorithm: str, total_episodes: int):
        """Start training monitoring."""
        # First clear any existing training data
        self._init_file(self.training_file, is_training=True)
        self._init_file(self.testing_file, is_training=False)
        
        # Then set up new training session
        data = {
            'episode': 0,
            'total_episodes': total_episodes,
            'reward': 0.0,
            'std_reward': 0.0,
            'mean_energy': 0.0,
            'mean_violations': 0.0,
            'scenario': 'N/A',
            'is_training': True,
            'is_testing': False,
            'algorithm': algorithm,
            'active': True,
            'training_start_time': datetime.now().isoformat(),
            'scenarios': {},
            'completed': False
        }
        self._safe_write_json(self.training_file, data)
    
    def start_testing(self, algorithm: str, total_episodes: int):
        """Start testing monitoring."""
        # First clear any existing testing data
        self._init_file(self.testing_file, is_training=False)
        self._init_file(self.training_file, is_training=True)
        
        # Then set up new testing session
        data = {
            'episode': 0,
            'total_episodes': total_episodes,
            'reward': 0.0,
            'std_reward': 0.0,
            'mean_energy': 0.0,
            'mean_violations': 0.0,
            'scenario': 'N/A',
            'is_training': False,
            'is_testing': True,
            'algorithm': algorithm,
            'active': True,
            'testing_start_time': datetime.now().isoformat(),
            'scenarios': {},
            'completed': False
        }
        self._safe_write_json(self.testing_file, data)
    
    def update_progress(self, episode: int, reward: float, total_episodes: int = None,
                       scenario: str = None, mean_energy: float = None,
                       mean_violations: float = None, std_reward: float = None,
                       is_training: bool = True):
        """Update progress metrics."""
        # Determine which file to update
        filepath = self.training_file if is_training else self.testing_file
        
        try:
            # Read current data with file locking
            data = self._safe_read_json(filepath)
            
            # Get previous scenario if it exists
            prev_scenario = data.get('scenario', 'N/A')
            
            # Update metrics
            data['episode'] = episode
            if total_episodes is not None:
                data['total_episodes'] = total_episodes
            data['reward'] = reward
            if std_reward is not None:
                data['std_reward'] = std_reward
            if mean_energy is not None:
                data['mean_energy'] = mean_energy
            if mean_violations is not None:
                data['mean_violations'] = mean_violations
            
            # Handle scenario changes and completion
            if scenario is not None:
                # If scenario changed and previous scenario was valid, store it as completed
                if prev_scenario != 'N/A' and prev_scenario != scenario:
                    data['scenarios'][prev_scenario] = {
                        'reward': data.get('reward', 0),
                        'std_reward': data.get('std_reward', 0),
                        'mean_energy': data.get('mean_energy', 0),
                        'mean_violations': data.get('mean_violations', 0)
                    }
                
                # Update current scenario
                data['scenario'] = scenario
            
            # Write updated data with file locking
            self._safe_write_json(filepath, data)
                
        except Exception as e:
            print(f"Error updating progress: {str(e)}")
    
    def end_training(self):
        """End training monitoring."""
        try:
            data = self._safe_read_json(self.training_file)
            data['is_training'] = False
            data['active'] = False
            data['completed'] = True
            self._safe_write_json(self.training_file, data)
        except Exception as e:
            print(f"Error ending training: {str(e)}")
    
    def end_testing(self):
        """End testing monitoring."""
        try:
            data = self._safe_read_json(self.testing_file)
            data['is_testing'] = False
            data['active'] = False
            data['completed'] = True
            self._safe_write_json(self.testing_file, data)
        except Exception as e:
            print(f"Error ending testing: {str(e)}")
    
    def clear_all(self):
        """Clear all monitoring data."""
        self._init_file(self.training_file, is_training=True)
        self._init_file(self.testing_file, is_training=False) 