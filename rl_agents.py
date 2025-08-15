import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import threading
import copy
from gymnasium import spaces
import time
import torch.multiprocessing as mp
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt


class BaseNetwork(nn.Module):
    """Base neural network architecture for all agents."""
    
    def __init__(self, state_dims: Dict[str, int], hidden_dims: List[int]):
        """
        Initialize the network.
        
        Args:
            state_dims: Dictionary of state dimensions for each input component
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Network using device: {self.device}")
        
        # Calculate total input dimension
        self.input_dim = 0
        
        # Building features (8 features per building)
        self.input_dim += state_dims['building']
        
        # Weather features (4 total: temperature, humidity, diffuse solar, direct solar)
        self.input_dim += 4
        
        # Carbon intensity (1 feature)
        self.input_dim += 1
        
        # Time features (2 features: hour, day_of_week)
        self.input_dim += 2
        
        # Create layers
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        
        # Move network to device
        self.to(self.device)
        #print(f"Network architecture: {self.layers}")
        
    def _process_state(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to flat tensor with special case for batched tensors."""
        # Special case for pre-processed batched tensors from _batch_forward
        if "_batch_tensor" in state:
            return state["_batch_tensor"]  # Already a tensor
            
        features = []
        
        # Process building states
        for building_data in state['buildings'].values():
            for value in building_data.values():
                if isinstance(value, np.ndarray):
                    features.extend(value.flatten())
                else:
                    features.append(value)
            
        # Process weather features
        features.extend(state['weather']['temperature'].flatten())
        features.extend(state['weather']['humidity'].flatten())
        features.extend(state['weather']['solar_irradiance'].flatten())
        
        # Process carbon intensity
        features.extend(state['carbon_intensity'].flatten())
        
        # Process time features
        features.extend(state['time_features'].flatten())
        
        # Convert to tensor and move to device
        return torch.FloatTensor(features).to(self.device)


class DuelingHead(nn.Module):
    """Dueling network architecture for DQN."""
    
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage (using the dueling architecture formula)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQNNetwork(BaseNetwork):
    """Deep Q-Network for discrete action spaces with enhanced GPU utilization."""
    
    def __init__(self, state_dims: Dict[str, int], action_dims: Dict[str, int], hidden_dims: List[int] = [512, 256]):
        """
        Initialize the DQN network with moderate capacity but optimized for speed.
        
        Args:
            state_dims: Dictionary of state dimensions
            action_dims: Dictionary mapping building IDs to number of actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__(state_dims, hidden_dims)
        
        # Enable mixed precision training
        self.scaler = torch.amp.GradScaler()
        
        # Simpler architecture with layer norm
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU()
        )
        
        # Reduced dropout
        self.dropout = nn.Dropout(0.1)  # Reduced from 0.15
        
        # Output heads without dueling unless really needed
        self.output_heads = nn.ModuleDict({
            building_id: nn.Linear(hidden_dims[-1], n_actions)
            for building_id, n_actions in action_dims.items()
        })
        
        # Move all components to device
        self.output_heads.to(self.device)
        
        # Create CUDA events for timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)


    def forward(self, state: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass using dueling architecture and convolutional features."""
        # Get base features
        x = self._process_state(state) if not isinstance(state, dict) or "_batch_tensor" not in state else state["_batch_tensor"]
        
        # Add convolutional processing
        # Reshape for conv1d [batch, channels, features]
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        # Process through base layers
        features = self.layers(x)
        
        # Apply dropout and normalization
        features = self.dropout(features)
        # features = self.layer_norm(features) # Removed layer_norm
        
        # Dueling architecture for each building
        results = {}
        for building_id, dueling_head in self.output_heads.items():
            q_values = dueling_head(features)
            results[building_id] = q_values
            
        return results


class DQNAgent:
    """DQN agent with experience replay and target network."""
    
    def __init__(self,
                 state_dims: Dict[str, int],
                 action_dims: Dict[str, int],
                 learning_rate: float = 5e-4,  # Increased from 2e-4
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.997,  # Slower decay
                 buffer_size: int = 100000,  # Increased buffer
                 batch_size: int = 256,  # Reduced from 1024
                 target_update: int = 100,  # Less frequent updates
                 n_envs: int = 16,
                 patience: int = 5000,  # Added patience parameter
                 min_reward_threshold: float = 10.0,  # Added minimum reward threshold
                 min_episodes: int = 100):  # Added minimum episodes before early stopping
        """Initialize the DQN agent."""
        self.action_dims = action_dims
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.training_steps = 0
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        
        # Early stopping parameters
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.min_episodes = min_episodes
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.last_best_update = 0
        self.total_episodes = 0
        
        # Set device and enable mixed precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler()
        
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            
            # Create CUDA streams for parallel processing
            self.streams = [torch.cuda.Stream() for _ in range(2)]
        
        # Networks
        self.policy_net = DQNNetwork(state_dims, action_dims).to(self.device)
        self.target_net = DQNNetwork(state_dims, action_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Disable gradient calculation for target network
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Optimizer with reduced weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # Reduced from 1e-4
        )
        
        # Single unified buffer instead of multiple buffers
        self.buffer = deque(maxlen=buffer_size)
        
        # Enable gradient caching for faster updates
        for p in self.policy_net.parameters():
            p.register_hook(lambda grad: torch.nan_to_num(grad, 0.0))

        if torch.cuda.is_available():
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
    def select_action(self, state: Dict, training: bool = True) -> Dict[str, int]:
        """Select action using epsilon-greedy policy for each building."""
        actions = {}
        # Evaluate all Q-values in a single forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                q_dict = self.policy_net({"_batch_tensor": self.policy_net._process_state(state).unsqueeze(0)})
                
            for building_id, n_actions in self.action_dims.items():
                if training and random.random() < self.epsilon:
                    actions[building_id] = random.randint(0, n_actions - 1)
                else:
                    q_values = q_dict[building_id]
                    actions[building_id] = q_values.cpu().argmax().item()
        return actions
        
    def store_transition(self, state: Dict, action: Dict[str, int], reward: float, next_state: Dict, done: bool):
        """Store transition with TD error prioritization."""
        # Calculate TD error for prioritization
        with torch.no_grad():
            state_tensor = self.policy_net._process_state(state)
            next_state_tensor = self.policy_net._process_state(next_state)
            current_q_dict = self.policy_net({"_batch_tensor": state_tensor.unsqueeze(0)})
            next_q_dict = self.target_net({"_batch_tensor": next_state_tensor.unsqueeze(0)})
            
            # Calculate TD error across all buildings
            td_error = 0.0
            for building_id in self.action_dims.keys():
                # Get current Q value for the action taken
                current_q = current_q_dict[building_id][0, action[building_id]].item()
                # Get max Q value for next state
                next_q = next_q_dict[building_id][0].max().item()
                # Add to total TD error
                td_error += abs(reward + self.gamma * next_q * (1 - done) - current_q)
                
            # Average TD error across buildings
            td_error = td_error / len(self.action_dims)
            
            self.buffer.append((state_tensor, action, float(reward), next_state_tensor, bool(done), td_error))
        
    def update(self) -> Optional[float]:
        """Update policy network using mixed precision and parallel processing."""
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample with priority, but handle case when buffer is smaller than 2*batch_size
        sample_size = min(self.batch_size * 2, len(self.buffer))
        batch = sorted(random.sample(self.buffer, sample_size), 
                      key=lambda x: x[5],  # sort by TD error
                      reverse=True)[:self.batch_size]
        
        # Efficient batch processing
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        
        # Process in single batch
        state_batch = torch.stack(states).to(self.device)
        next_state_batch = torch.stack(next_states).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        action_tensors = {}
        for b_id in self.action_dims.keys():
            action_tensors[b_id] = torch.LongTensor([a[b_id] for a in actions]).to(self.device)
        
        # Use mixed precision for forward passes
        with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            # Get current Q values
            current_q_batch = self.policy_net({"_batch_tensor": state_batch})
            
            with torch.no_grad():
                # Get next Q values
                next_q_batch = self.target_net({"_batch_tensor": next_state_batch})
            
            # Compute loss for each building
            batch_loss = torch.tensor(0.0, device=self.device)
            
            for b_id in self.action_dims.keys():
                q_cur = current_q_batch[b_id].gather(1, action_tensors[b_id].unsqueeze(1)).squeeze(1)
                q_next = next_q_batch[b_id].max(1)[0].detach()
                
                # Compute target Q values
                target_q = reward_batch + self.gamma * q_next * (1 - done_batch)
                
                # Compute loss
                loss = F.smooth_l1_loss(q_cur, target_q)
                batch_loss += loss
            
            batch_loss = batch_loss / len(self.action_dims)
        
        # Scale loss and backward pass
        self.scaler.scale(batch_loss).backward()
        total_loss = batch_loss.item()
        
        # Optimizer step with gradient scaling
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return total_loss


class ActorCritic(BaseNetwork):
    """Combined actor-critic network for PPO and A3C."""
    
    def __init__(self, state_dims: Dict[str, int], n_actions: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dims: Dictionary of state dimensions
            n_actions: Number of discrete actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__(state_dims, hidden_dims)
        
        # Actor (policy) head with layer normalization
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.Linear(hidden_dims[-1] // 2, n_actions)
        )
        
        # Critic (value) head with layer normalization
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Move actor and critic to the configured device
        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value."""
        x = self._process_state(state)
        features = self.layers(x)
        
        # Get logits from actor
        logits = self.actor(features)
        
        # Apply softmax with numerical stability
        policy = F.softmax(logits, dim=-1)
        
        # Ensure valid probability distribution
        policy = torch.clamp(policy, min=1e-6, max=1.0)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        return policy, self.critic(features)


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self,
                 state_dims: Dict[str, int],
                 n_actions: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 patience: int = 5000,  # Added patience parameter
                 min_reward_threshold: float = 10.0,  # Added minimum reward threshold
                 min_episodes: int = 100):  # Added minimum episodes before early stopping
        """
        Initialize the PPO agent.
        
        Args:
            state_dims: Dictionary of state dimensions
            n_actions: Number of discrete actions
            learning_rate: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            patience: Number of episodes to wait for improvement before early stopping
            min_reward_threshold: Minimum reward threshold for early stopping
            min_episodes: Minimum number of episodes before early stopping
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Early stopping parameters
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.min_episodes = min_episodes
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.last_best_update = 0
        self.total_episodes = 0
        
        # Networks
        self.network = ActorCritic(state_dims, n_actions)
        self.device = self.network.device
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def select_action(self, state: Dict, training: bool = True) -> Tuple[int, float, torch.Tensor]:
        """Select action using current policy."""
        with torch.no_grad():
            policy, value = self.network(state)
            if training:
                action = torch.multinomial(policy, 1).item()
            else:
                action = policy.argmax().item()
            return action, value.item(), policy
            
    def update(self, 
              states: List[Dict],
              actions: List[int],
              rewards: List[float],
              values: List[float],
              old_policies: List[torch.Tensor],
              next_states: List[Dict],
              dones: List[bool]) -> Dict[str, float]:
        """Update policy using PPO objective."""
        # Convert to tensors
        state_tensor = torch.stack([self.network._process_state(s) for s in states]).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        value_tensor = torch.FloatTensor(values).to(self.device)
        old_policy_tensor = torch.stack(old_policies).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            next_values = torch.cat([self.network(s)[1] for s in next_states])
            advantages = reward_tensor + self.gamma * next_values * (1 - done_tensor) - value_tensor
            returns = advantages + value_tensor
            
        # PPO update
        for _ in range(10):  # Multiple epochs
            policy, value = self.network({"_batch_tensor": state_tensor})
            
            # Compute policy ratio and clipped objective
            ratio = torch.exp(torch.log(policy.gather(1, action_tensor.unsqueeze(1))) - 
                            torch.log(old_policy_tensor.gather(1, action_tensor.unsqueeze(1))))
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages.unsqueeze(1)
            policy_loss = -torch.min(ratio * advantages.unsqueeze(1), clip_adv).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(value.squeeze(), returns)
            
            # Compute entropy bonus
            entropy = -(policy * torch.log(policy)).sum(1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


class A3CWorker:
    """Worker process for A3C."""
    
    def __init__(self,
                 worker_id: int,
                 state_dims: Dict[str, int],
                 n_actions: int,
                 global_network: ActorCritic,
                 optimizer: torch.optim.Optimizer,
                 env_creator,
                 device: torch.device,
                 gamma: float = 0.99,
                 t_max: int = 20,
                 tau: float = 0.95,
                 total_steps = None,
                 best_reward = None,
                 episode_count = None,
                 rewards_sum = None,
                 entropy_coef: float = 0.01):
        """Initialize A3C worker."""
        self.worker_id = worker_id
        self.gamma = gamma
        self.t_max = t_max
        self.tau = tau
        self.device = device
        
        # Create local network and environment
        self.local_network = ActorCritic(state_dims, n_actions).to(device)
        self.env = env_creator()
        
        # Store references to global network and optimizer
        self.global_network = global_network
        self.optimizer = optimizer
        
        # Shared counters
        self.total_steps = total_steps
        self.best_reward = best_reward
        self.episode_count = episode_count
        self.rewards_sum = rewards_sum
        self.entropy_coef = entropy_coef
        
        # Local stats
        self.episode_reward = 0
        self.running_reward = 0
        
        # Process control
        self._running = mp.Value('b', True)
        
        # Initialize weights
        self.sync_with_global()
        
    def stop(self):
        """Stop the worker process."""
        if hasattr(self, '_running'):
            with self._running.get_lock():
                self._running.value = False
                
    def sync_with_global(self):
        """Synchronize local network with global network."""
        try:
            self.local_network.load_state_dict(self.global_network.state_dict())
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in sync_with_global: {str(e)}")
            
    def update_counters(self, steps_delta: int, reward: float):
        """Update shared counters with thread safety."""
        if self.total_steps is not None:
            with self.total_steps.get_lock():
                self.total_steps.value += steps_delta
                
        if self.best_reward is not None and reward > self.best_reward.value:
            with self.best_reward.get_lock():
                if reward > self.best_reward.value:  # Check again after acquiring lock
                    self.best_reward.value = reward
                    
        if self.episode_count is not None:
            with self.episode_count.get_lock():
                self.episode_count.value += 1
                
        if self.rewards_sum is not None:
            with self.rewards_sum.get_lock():
                self.rewards_sum.value += reward
            
    def compute_returns_and_advantages(self, rewards: List[float], values: List[float], next_value: float, done: bool) -> Tuple[List[float], List[float]]:
        """Compute returns and advantages using GAE."""
        returns = []
        advantages = []
        advantage = 0
        next_value = next_value * (1 - int(done))
        
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            advantage = td_error + self.gamma * self.tau * advantage
            next_value = v
            
            returns.insert(0, advantage + v)
            advantages.insert(0, advantage)
            
        return returns, advantages
        
    def get_action(self, state: Dict) -> Tuple[int, float, torch.Tensor]:
        """Select action using current policy with proper error handling."""
        try:
            with torch.no_grad():
                state_tensor = self.local_network._process_state(state).to(self.device)
                policy, value = self.local_network({"_batch_tensor": state_tensor.unsqueeze(0)})
                
                # Verify policy is valid
                if torch.isnan(policy).any() or torch.isinf(policy).any():
                    print(f"Worker {self.worker_id}: Invalid policy values detected, using uniform distribution")
                    policy = torch.ones_like(policy) / policy.shape[-1]
                
                # Ensure policy sums to 1
                policy = F.normalize(policy, p=1, dim=-1)
                
                # Create categorical distribution
                try:
                    action_dist = torch.distributions.Categorical(probs=policy)
                    action = action_dist.sample().item()
                except ValueError as e:
                    print(f"Worker {self.worker_id}: Error creating distribution, using random action")
                    action = torch.randint(0, policy.shape[-1], (1,)).item()
                
                return action, value.item(), policy
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in get_action: {str(e)}")
            # Return random action as fallback
            n_actions = policy.shape[-1] if 'policy' in locals() else self.local_network.actor[-1].out_features
            return (
                torch.randint(0, n_actions, (1,)).item(),
                0.0,
                torch.ones(1, n_actions).to(self.device) / n_actions
            )
            
    def run(self):
        """Run worker process."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                n_episodes = 0
                max_episodes = 10000  # Add a reasonable limit
                
                while n_episodes < max_episodes:
                    # Check if we should stop
                    with self._running.get_lock():
                        if not self._running.value:
                            break
                    
                    self.sync_with_global()
                    
                    # Storage for episode experience
                    states, actions, rewards, values, policies = [], [], [], [], []
                    
                    # Reset environment
                    state, _ = self.env.reset()
                    done = False
                    self.episode_reward = 0
                    steps_this_episode = 0
                    
                    # Run for t_max steps or until episode ends
                    for t in range(self.t_max):
                        # Check if we should stop
                        with self._running.get_lock():
                            if not self._running.value:
                                break
                        
                        # Get action using stable action selection
                        action, value, policy = self.get_action(state)
                        
                        # Take action in environment
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        
                        # Store experience
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        values.append(value)
                        policies.append(policy)
                        
                        self.episode_reward += reward
                        steps_this_episode += 1
                        
                        if done:
                            break
                            
                        state = next_state
                    
                    # Update shared counters
                    self.update_counters(steps_this_episode, self.episode_reward)
                    
                    if len(states) > 0:  # Only update if we have collected some experience
                        try:
                            # Get final value (0 if done, otherwise bootstrap)
                            final_value = 0 if done else self.get_action(next_state)[1]
                            
                            # Compute returns and advantages
                            returns, advantages = self.compute_returns_and_advantages(rewards, values, final_value, done)
                            
                            # Convert to tensors
                            state_tensor = torch.stack([self.local_network._process_state(s) for s in states]).to(self.device)
                            action_tensor = torch.LongTensor(actions).to(self.device)
                            return_tensor = torch.FloatTensor(returns).to(self.device)
                            advantage_tensor = torch.FloatTensor(advantages).to(self.device)
                            
                            # Get current policy and value
                            policy, value = self.local_network({"_batch_tensor": state_tensor})
                            
                            # Verify policy is valid
                            if torch.isnan(policy).any() or torch.isinf(policy).any():
                                print(f"Worker {self.worker_id}: Invalid policy in update, skipping")
                                continue
                            
                            # Compute actor loss with numerical stability
                            log_probs = torch.log(policy.gather(1, action_tensor.unsqueeze(1)) + 1e-10)
                            actor_loss = -(log_probs * advantage_tensor.unsqueeze(1)).mean()
                            
                            # Compute critic loss
                            critic_loss = F.mse_loss(value.squeeze(), return_tensor)
                            
                            # Compute entropy bonus with numerical stability
                            entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=-1).mean()
                            
                            # Total loss
                            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy # Use worker's entropy_coef
                            
                            # Verify loss is valid
                            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                                # Compute gradients
                                self.optimizer.zero_grad()
                                total_loss.backward()
                                
                                # Clip gradients
                                torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), max_norm=40.0)
                                
                                # Ensure global_network is on the same device
                                if next(self.global_network.parameters()).device != self.device:
                                    self.global_network.to(self.device)
                                
                                # Copy gradients to global network
                                for global_param, local_param in zip(self.global_network.parameters(), self.local_network.parameters()):
                                    if global_param.grad is None:
                                        global_param.grad = local_param.grad.clone()
                                    else:
                                        global_param.grad += local_param.grad.clone()
                                
                                # Update global network
                                self.optimizer.step()
                            else:
                                print(f"Worker {self.worker_id}: Invalid loss value detected, skipping update")
                                
                        except Exception as e:
                            print(f"Worker {self.worker_id}: Error in update step: {str(e)}")
                            continue
                    
                    if done:
                        state, _ = self.env.reset()
                        n_episodes += 1
                        
                        # Print episode stats occasionally
                        if n_episodes % 10 == 0:
                            print(f"\nWorker {self.worker_id} - Episode {n_episodes}, Reward: {self.episode_reward:.2f}")
                        
            except Exception as e:
                print(f"Worker {self.worker_id}: Fatal error: {str(e)}")
                raise


def _run_worker_with_warnings_suppressed(worker):
    """Helper function to run worker with warnings suppressed."""
    worker.run()

class A3CAgent:
    """Asynchronous Advantage Actor-Critic agent with improved CUDA support."""
    
    def __init__(self,
                 state_dims: Dict[str, int],
                 n_actions: int,
                 env_creator,
                 n_workers: int = 4,
                 learning_rate: float = 3e-4,  # Increased learning rate
                 gamma: float = 0.99):
        """Initialize A3C agent."""
        # Enable multiprocessing method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Create global network and optimizer
        self.global_network = ActorCritic(state_dims, n_actions)
        
        # Move model to shared memory
        self.global_network.share_memory()
        
        # Create optimizer after moving model to shared memory
        self.optimizer = torch.optim.Adam(self.global_network.parameters(), lr=learning_rate, eps=1e-5)
        
        # Create shared counters and metrics
        self.total_steps = mp.Value('i', 0)
        self.best_reward = mp.Value('d', float('-inf'))
        self.episode_count = mp.Value('i', 0)
        self.rewards_sum = mp.Value('d', 0.0)
        self.no_improvement_count = mp.Value('i', 0)
        
        # Create a list to store rewards history
        self.rewards_history = []
        
        # Create workers
        self.workers = [
            A3CWorker(
                worker_id=i,
                state_dims=state_dims,
                n_actions=n_actions,
                global_network=self.global_network,
                optimizer=self.optimizer,
                env_creator=env_creator,
                device=self.device,
                gamma=gamma,
                t_max=20,
                tau=0.95,
                total_steps=self.total_steps,
                best_reward=self.best_reward,
                episode_count=self.episode_count,
                rewards_sum=self.rewards_sum,
                entropy_coef=0.01 * max(1.0 - i/(n_workers*2), 0.01)
            )
            for i in range(n_workers)
        ]
        
        self.n_workers = n_workers
        #(f"Created {n_workers} A3C workers")
        
        # Training control
        self.training = False
        
    def train(self, eval_interval: int = 10, save_dir: str = 'models', 
              max_episodes: int = 1000,
              patience: int = 5000,
              min_reward_threshold: float = 10,
              min_episodes: int = 100) -> Dict:  # Added minimum episodes
        """Start all worker threads and monitor training progress."""
        print("Starting A3C training...")
        self.training = True
        
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize training monitor
            from dashboard.utils import TrainingMonitor
            monitor = TrainingMonitor()
            
            # Start all workers
            processes = []
            for worker in self.workers:
                p = mp.Process(target=_run_worker_with_warnings_suppressed, args=(worker,))
                p.start()
                processes.append(p)
            
            # Monitor training progress
            last_eval = 0
            best_eval_reward = float('-inf')
            no_improvement_count = 0
            last_best_update = 0
            last_episode_count = 0
            self.rewards_history = []
            
            # Initialize progress at 0
            monitor.update_progress(
                episode=0,
                reward=0.0,
                total_episodes=max_episodes,
                mean_energy=0.0,
                mean_violations=0.0,
                is_training=True
            )
            
            while self.training:
                active_processes = sum(p.is_alive() for p in processes)
                if active_processes == 0:
                    #print("\nAll workers finished")
                    break
                    
                # Get current metrics
                with self.episode_count.get_lock():
                    episodes = self.episode_count.value
                with self.total_steps.get_lock():
                    steps = self.total_steps.value
                with self.rewards_sum.get_lock():
                    rewards_sum = self.rewards_sum.value
                with self.best_reward.get_lock():
                    best_reward = self.best_reward.value
                
                # Calculate average reward for new episodes
                if episodes > last_episode_count:
                    avg_reward = rewards_sum / episodes
                    # Add rewards for each new episode
                    for _ in range(episodes - last_episode_count):
                        self.rewards_history.append(avg_reward)
                    last_episode_count = episodes
                    
                    # Update training monitor with current progress
                    monitor.update_progress(
                        episode=episodes,
                        reward=avg_reward,
                        total_episodes=max_episodes,
                        mean_energy=0.0,  # These could be updated with actual values if available
                        mean_violations=0.0,
                        is_training=True
                    )
                
                # Early stopping checks - only after minimum episodes
                if episodes >= min_episodes:
                    if episodes >= max_episodes:
                        print(f"\nReached maximum episodes limit ({max_episodes})")
                        break
                        
                    if best_reward >= min_reward_threshold:
                        print(f"\nReached target reward threshold: {best_reward:.2f} after {episodes} episodes")
                        break
                
                # Evaluation and saving
                if episodes > 0 and episodes - last_eval >= eval_interval:
                    print(f"\nEpisode {episodes}")
                    print(f"Average reward: {avg_reward:.2f}")
                    print(f"Best reward: {best_reward:.2f}")
                    print(f"Total steps: {steps}")
                    
                    # Check for improvement
                    if best_reward > best_eval_reward:
                        best_eval_reward = best_reward
                        self.save_model('a3c', save_dir, is_best=True)
                        print(f"New best model saved with reward: {best_reward:.2f}")
                        no_improvement_count = 0
                        last_best_update = episodes
                    else:
                        no_improvement_count += 1
                        episodes_since_improvement = episodes - last_best_update
                        print(f"Episodes without improvement: {episodes_since_improvement}")
                        
                        # Early stopping check - only after minimum episodes
                        if episodes >= min_episodes and episodes_since_improvement >= patience:
                            print(f"\nStopping training - No improvement for {patience} episodes")
                            break
                    
                    last_eval = episodes
                else:
                    # Print progress
                    print(f"\rEpisode {episodes}, Steps: {steps}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}", end="")
                
                # Small sleep to prevent busy waiting
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.training = False
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.training = False
        finally:
            self.stop(processes)
            
            # Save final results
            results = {
                'rewards': self.rewards_history,
                'best_reward': float(best_eval_reward),
                'total_steps': int(self.total_steps.value),
                'total_episodes': int(self.episode_count.value),
                'stopped_early': no_improvement_count >= patience,
                'reason': 'patience' if no_improvement_count >= patience else 
                        'max_episodes' if episodes >= max_episodes else
                        'target_reached' if best_reward >= min_reward_threshold else
                        'completed',
                'training_duration': len(self.rewards_history)
            }
            
            # Save training progress plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards_history)
            plt.title('A3C Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            # Add moving average
            if len(self.rewards_history) > 10:
                window_size = min(10, len(self.rewards_history) // 5)
                moving_avg = np.convolve(self.rewards_history, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(self.rewards_history)), moving_avg, 'r-', linewidth=2)
                plt.legend(['Episode Reward', f'{window_size}-Episode Moving Avg'])
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(save_dir, f'a3c_training_progress_{timestamp}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"\nTraining progress plot saved to {plot_path}")
            
            # Save results to JSON
            results_path = os.path.join(save_dir, 'a3c_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f"Training completed. Results saved to {results_path}")
            
            return results
            
    def stop(self, processes=None):
        """Stop all worker processes."""
        print("\nStopping A3C training...")
        self.training = False
        
        if processes:
            try:
                # Terminate all processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5.0)
                        
                #print("All workers stopped")
                
            except Exception as e:
                print(f"Error stopping workers: {str(e)}")
                
    def save_model(self, algorithm: str, save_dir: str, is_best: bool = False):
        """
        Save model weights and configuration.
        
        Args:
            algorithm: Algorithm name ('a3c')
            save_dir: Directory to save the model
            is_best: Whether this is the best model so far
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model weights with timestamp
        model_path = os.path.join(save_dir, f'{algorithm}_model_{timestamp}.pt')
        torch.save(self.global_network.state_dict(), model_path)
        
        # Save configuration with timestamp
        config = {
            'algorithm': algorithm,
            'timestamp': timestamp,
            'model_path': model_path,
            'total_steps': int(self.total_steps.value),
            'best_reward': float(self.best_reward.value),
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
            torch.save(self.global_network.state_dict(), best_model_path)
            
            # Save best model config
            best_config = {
                'algorithm': algorithm,
                'timestamp': timestamp,
                'model_path': best_model_path,
                'total_steps': int(self.total_steps.value),
                'best_reward': float(self.best_reward.value),
                'is_best': True,
                'original_model_path': model_path
            }
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=4)
            
    def select_action(self, state: Dict, training: bool = False) -> int:
        """Select action using global network."""
        self.global_network.eval()
        with torch.no_grad():
            state_tensor = self.global_network._process_state(state).to(self.device)
            policy, _ = self.global_network({"_batch_tensor": state_tensor.unsqueeze(0)})
            if training:
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample().item()
            else:
                action = policy.argmax().item()
        return action 