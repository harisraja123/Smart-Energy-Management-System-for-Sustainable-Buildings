import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import traceback  # Add traceback for better error logging

def load_test_results(algorithm: str) -> dict:
    """Load test results for a specific algorithm."""
    try:
        # First try to load individual algorithm results
        filepath = os.path.join('models', f'{algorithm}_test_results.json')
        st.write(f"Looking for individual results at: {filepath}")  # Debug log
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                st.write(f"Loaded individual results for {algorithm}")  # Debug log
                return data
                
        # If individual results don't exist, try loading from combined results
        combined_filepath = os.path.join('models', 'all_test_results.json')
        st.write(f"Looking for combined results at: {combined_filepath}")  # Debug log
        if os.path.exists(combined_filepath):
            with open(combined_filepath, 'r') as f:
                all_results = json.load(f)
                if algorithm in all_results:
                    # Calculate aggregate metrics across scenarios
                    scenarios = all_results[algorithm]
                    st.write(f"Found {algorithm} in combined results with scenarios: {list(scenarios.keys())}")  # Debug log
                    try:
                        mean_reward = np.mean([r['mean_reward'] for r in scenarios.values()])
                        std_reward = np.mean([r['std_reward'] for r in scenarios.values()])
                        mean_energy = np.mean([r['mean_energy'] for r in scenarios.values()])
                        mean_violations = np.mean([r['mean_violations'] for r in scenarios.values()])
                        
                        return {
                            'mean_reward': float(mean_reward),
                            'std_reward': float(std_reward),
                            'mean_energy': float(mean_energy),
                            'mean_violations': float(mean_violations),
                            'scenarios': scenarios
                        }
                    except Exception as calc_error:
                        st.error(f"Error calculating metrics for {algorithm}: {str(calc_error)}")
                        st.write("Scenario data:", scenarios)  # Debug log
                        st.write(traceback.format_exc())  # Show full traceback
                else:
                    st.write(f"Algorithm {algorithm} not found in combined results. Available: {list(all_results.keys())}")  # Debug log
    except Exception as e:
        st.error(f"Error loading test results for {algorithm}: {str(e)}")
        st.write(traceback.format_exc())  # Show full traceback
    return None

def create_comparison_chart(results: dict) -> go.Figure:
    """Create a comparison chart for different algorithms."""
    try:
        # Extract data for plotting
        algorithms = []
        mean_rewards = []
        std_rewards = []
        mean_energy = []
        violations = []
        
        for algo, data in results.items():
            if isinstance(data, dict):  # Check if data is valid
                algorithms.append(algo.upper())
                mean_rewards.append(data.get('mean_reward', 0))
                std_rewards.append(data.get('std_reward', 0))
                mean_energy.append(data.get('mean_energy', 0))
                violations.append(data.get('mean_violations', 0))
        
        if not algorithms:  # If no valid data was found
            st.error("No valid algorithm data found for plotting")
            return None
            
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add reward bars with error bars
        fig.add_trace(go.Bar(
            name='Mean Reward',
            x=algorithms,
            y=mean_rewards,
            error_y=dict(
                type='data',
                array=std_rewards,
                visible=True
            ),
            marker_color='rgb(55, 83, 109)'
        ))
        
        # Add energy consumption line
        fig.add_trace(go.Scatter(
            name='Mean Energy (kWh)',
            x=algorithms,
            y=mean_energy,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(color='rgb(26, 118, 255)', width=2),
            yaxis='y2'
        ))
        
        # Add comfort violations line
        fig.add_trace(go.Scatter(
            name='Comfort Violations',
            x=algorithms,
            y=violations,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(color='rgb(219, 64, 82)', width=2),
            yaxis='y3'
        ))
        
        # Update layout with correct axis positioning
        fig.update_layout(
            title='Algorithm Performance Comparison',
            xaxis=dict(
                title='Algorithm',
                domain=[0, 0.8]  # Make space for multiple y-axes
            ),
            yaxis=dict(
                title=dict(
                    text='Mean Reward',
                    font=dict(color='rgb(55, 83, 109)')
                ),
                tickfont=dict(color='rgb(55, 83, 109)')
            ),
            yaxis2=dict(
                title=dict(
                    text='Mean Energy (kWh)',
                    font=dict(color='rgb(26, 118, 255)')
                ),
                tickfont=dict(color='rgb(26, 118, 255)'),
                anchor='free',
                overlaying='y',
                side='right',
                position=0.85  # Position within valid range [0,1]
            ),
            yaxis3=dict(
                title=dict(
                    text='Comfort Violations',
                    font=dict(color='rgb(219, 64, 82)')
                ),
                tickfont=dict(color='rgb(219, 64, 82)'),
                anchor='free',
                overlaying='y',
                side='right',
                position=0.95  # Position within valid range [0,1]
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Adjust margins to ensure all axes are visible
            margin=dict(r=100, l=50, t=100, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        st.write(traceback.format_exc())  # Show full traceback
        return None

def show_comparison_results():
    """Show comparison of results between different algorithms."""
    st.title("Results Comparison")
    
    try:
        # Load results for each algorithm
        algorithms = ['dqn', 'ppo', 'a3c']
        results = {}
        
        for algo in algorithms:
            st.write(f"Loading results for {algo}...")  # Debug log
            data = load_test_results(algo)
            if data:
                results[algo] = data
                st.write(f"Successfully loaded {algo} results")  # Debug log
            else:
                st.write(f"No results found for {algo}")  # Debug log
        
        if not results:
            st.warning("No test results available. Run testing first to compare algorithms.")
            return
            
        st.write(f"Loaded results for algorithms: {list(results.keys())}")  # Debug log
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Overview", "Detailed Metrics"])
        
        with tab1:
            # Show comparison chart
            fig = create_comparison_chart(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Add summary table
            summary_data = []
            for algo, data in results.items():
                if isinstance(data, dict):
                    summary_data.append({
                        'Algorithm': algo.upper(),
                        'Mean Reward': f"{data.get('mean_reward', 0):.2f} ± {data.get('std_reward', 0):.2f}",
                        'Mean Energy': f"{data.get('mean_energy', 0):.2f} kWh",
                        'Comfort Violations': f"{data.get('mean_violations', 0):.2f}"
                    })
            
            if summary_data:
                st.subheader("Summary Table")
                df = pd.DataFrame(summary_data)
                st.table(df.set_index('Algorithm'))
        
        with tab2:
            # Show detailed metrics for each algorithm
            for algo, data in results.items():
                if isinstance(data, dict):
                    with st.expander(f"{algo.upper()} Detailed Metrics", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Reward",
                                f"{data.get('mean_reward', 0):.2f}",
                                f"±{data.get('std_reward', 0):.2f}"
                            )
                            
                        with col2:
                            st.metric(
                                "Mean Energy",
                                f"{data.get('mean_energy', 0):.2f} kWh"
                            )
                            
                        with col3:
                            st.metric(
                                "Comfort Violations",
                                data.get('mean_violations', 0)
                            )
                        
                        # Show scenario breakdown if available
                        if 'scenarios' in data:
                            st.subheader("Scenario Performance")
                            scenario_data = []
                            for scenario, metrics in data['scenarios'].items():
                                scenario_data.append({
                                    'Scenario': scenario.capitalize(),
                                    'Mean Reward': f"{metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}",
                                    'Mean Energy': f"{metrics.get('mean_energy', 0):.2f} kWh",
                                    'Comfort Violations': f"{metrics.get('mean_violations', 0):.2f}"
                                })
                            
                            if scenario_data:
                                df = pd.DataFrame(scenario_data)
                                st.table(df.set_index('Scenario'))
    except Exception as e:
        st.error(f"Error in comparison view: {str(e)}")
        st.write(traceback.format_exc())  # Show full traceback 