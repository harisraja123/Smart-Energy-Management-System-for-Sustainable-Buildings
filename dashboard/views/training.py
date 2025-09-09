import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time
import portalocker  # Cross-platform file locking

def load_training_data():
    """Load current training data."""
    max_retries = 3
    retry_delay = 0.1  # seconds
    
    for attempt in range(max_retries):
        try:
            filepath = os.path.join('dashboard', 'data', 'current_training.json')
            if os.path.exists(filepath):
                # Use portalocker for cross-platform file locking
                with portalocker.Lock(filepath, 'r', timeout=10) as f:
                    data = json.load(f)
                    
                    # Ensure we have all required fields with defaults
                    defaults = {
                        'episode': 0,
                        'total_episodes': 0,
                        'reward': 0.0,
                        'std_reward': 0.0,
                        'is_training': False,
                        'algorithm': 'N/A'
                    }
                    # Update data with any missing defaults
                    for key, value in defaults.items():
                        if key not in data:
                            data[key] = value
                    return data
            return None
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Error loading training data: Invalid JSON format. Retrying...")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Error loading training data: {str(e)}")
            return None

def format_time_elapsed(start_time_str):
    """Format elapsed time since training started."""
    if not start_time_str:
        return "Not started"
    
    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)
    seconds = int(elapsed.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def show_algorithm_section(algorithm: str, data: dict):
    """Show metrics and progress for a single algorithm."""
    st.subheader(f"{algorithm.upper()} Progress")
    
    # Training metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Reward", f"{data.get('reward', 0):.2f} ± {data.get('std_reward', 0):.2f}")
    
    with col2:
        progress = data.get('episode', 0) / max(data.get('total_episodes', 1), 1)
        st.progress(progress, text=f"Progress: {progress*100:.1f}%")

def show_training_progress():
    st.title("Training Monitor")
    
    # Add auto-refresh checkbox in sidebar
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    if auto_refresh:
        # Create a placeholder with changing content to force refresh
        refresh_placeholder = st.empty()
        current_time = time.time()
        refresh_placeholder.text(f"Refreshing... {current_time}")
        
        # Use st.empty() at the top level to force a rerun
        st.empty()
    
    # Load current training data
    data = load_training_data()
    
    if not data:
        st.warning("No training data available. Start training to see metrics.")
        return
    
    # Show overall status
    status = "🟢 Training" if data.get('is_training', False) else "⚪ Not Training"
    st.markdown(f"### Status: {status}")
    
    # Show last refresh time in sidebar
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown(f"**Time Elapsed:** {format_time_elapsed(data.get('training_start_time'))}")
    
    # Show training progress
    show_algorithm_section(data.get('algorithm', 'Unknown'), data)
    
    # Force rerun if auto-refresh is enabled
    if auto_refresh:
        time.sleep(5)
        st.rerun() 