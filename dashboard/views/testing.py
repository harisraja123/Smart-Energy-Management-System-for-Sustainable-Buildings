import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time

def load_testing_data():
    """Load current testing data."""
    try:
        filepath = os.path.join('dashboard', 'data', 'current_testing.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Ensure we have all required fields with defaults
                defaults = {
                    'episode': 0,
                    'total_episodes': 0,
                    'reward': 0.0,
                    'std_reward': 0.0,
                    'mean_energy': 0.0,
                    'mean_violations': 0.0,
                    'scenario': 'N/A',
                    'is_testing': False,
                    'algorithm': 'N/A',
                    'scenarios': {},  # Store completed scenarios here
                    'completed': False
                }
                # Update data with any missing defaults
                for key, value in defaults.items():
                    if key not in data:
                        data[key] = value
                return data
        return None
    except Exception as e:
        st.error(f"Error loading testing data: {str(e)}")
        return None

def format_time_elapsed(start_time_str):
    """Format elapsed time since testing started."""
    if not start_time_str:
        return "Not started"
    
    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)
    seconds = int(elapsed.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def show_scenario_metrics(scenario_name: str, data: dict, is_current: bool = False):
    """Show metrics for a scenario."""
    with st.expander(f"Scenario: {scenario_name}", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Reward", 
                    f"{data['reward']:.2f} ± {data['std_reward']:.2f}")
        with col2:
            st.metric("Mean Energy", f"{data['mean_energy']:.2f} kWh")
        with col3:
            st.metric("Comfort Violations", f"{data['mean_violations']:.2f}")
        
        if is_current:
            # Show progress bar for current scenario
            progress = data.get('episode', 0) / max(data.get('total_episodes', 1), 1)
            st.progress(progress, text=f"Progress: {progress*100:.1f}%")

def show_testing_progress():
    st.title("Testing Monitor")
    
    # Add auto-refresh checkbox in sidebar
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    if auto_refresh:
        # Create a placeholder with changing content to force refresh
        refresh_placeholder = st.empty()
        current_time = time.time()
        refresh_placeholder.text(f"Refreshing... {current_time}")
        
        # Use st.empty() at the top level to force a rerun
        st.empty()
    
    # Load current testing data
    data = load_testing_data()
    
    if not data:
        st.warning("No testing data available. Start testing to see metrics.")
        return
    
    # Show overall status
    status = "🟢 Testing" if data.get('is_testing', False) else "⚪ Not Testing"
    st.markdown(f"### Status: {status}")
    
    # Show last refresh time in sidebar
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    # Show time elapsed
    st.markdown(f"**Time Elapsed:** {format_time_elapsed(data.get('testing_start_time'))}")
    
    # Display current algorithm's results
    current_algo = data.get('algorithm', '')
    if current_algo:
        st.markdown(f"## {current_algo} Testing Results")
        
        # Show completed scenarios first
        completed_scenarios = data.get('scenarios', {})
        if completed_scenarios:
            st.markdown("### Completed Scenarios")
            for scenario_name, scenario_data in completed_scenarios.items():
                show_scenario_metrics(scenario_name, scenario_data)
        
        # Show current scenario if testing is ongoing and it's not already in completed scenarios
        current_scenario = data.get('scenario', '')
        if data.get('is_testing', False) and current_scenario and current_scenario != 'N/A' and current_scenario not in completed_scenarios:
            st.markdown("### Current Scenario")
            current_data = {
                'reward': data.get('reward', 0),
                'std_reward': data.get('std_reward', 0),
                'mean_energy': data.get('mean_energy', 0),
                'mean_violations': data.get('mean_violations', 0),
                'episode': data.get('episode', 0),
                'total_episodes': data.get('total_episodes', 1)
            }
            show_scenario_metrics(current_scenario, current_data, is_current=True)
        elif data.get('completed', False):
            st.success("Testing Completed")
        
        st.markdown("---")  # Separator between algorithms
    
    # Force rerun if auto-refresh is enabled
    if auto_refresh:
        time.sleep(5)
        st.rerun() 