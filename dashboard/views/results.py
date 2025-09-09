import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def load_all_results():
    """Load all results from the models directory"""
    results = {}
    try:
        for filename in os.listdir('models'):
            if filename.endswith('_results.json'):
                algorithm = filename.split('_')[0].upper()
                with open(os.path.join('models', filename), 'r') as f:
                    results[algorithm] = json.load(f)
        return results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def plot_reward_distribution(data):
    """Plot reward distribution using box plots"""
    if not data:
        return None
        
    # Prepare data for plotting
    plot_data = []
    for algo, results in data.items():
        rewards = results.get('rewards', [])
        plot_data.extend([{'Algorithm': algo, 'Reward': r} for r in rewards])
    
    df = pd.DataFrame(plot_data)
    
    fig = px.box(df, x='Algorithm', y='Reward',
                 title='Reward Distribution by Algorithm',
                 template='plotly_white')
    
    return fig

def plot_convergence_comparison(data):
    """Plot convergence comparison across algorithms"""
    if not data:
        return None
        
    fig = go.Figure()
    
    for algo, results in data.items():
        rewards = results.get('rewards', [])
        if rewards:
            # Calculate moving average
            window_size = min(10, len(rewards))
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(rewards) + 1)),
                y=moving_avg,
                name=algo,
                mode='lines',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Training Convergence Comparison',
        xaxis_title='Episode',
        yaxis_title='Moving Average Reward',
        template='plotly_white'
    )
    
    return fig

def calculate_statistics(data):
    """Calculate summary statistics for each algorithm"""
    stats = {}
    for algo, results in data.items():
        rewards = results.get('rewards', [])
        if rewards:
            stats[algo] = {
                'Mean Reward': np.mean(rewards),
                'Max Reward': np.max(rewards),
                'Min Reward': np.min(rewards),
                'Std Dev': np.std(rewards),
                'Episodes': len(rewards),
                'Convergence': len(rewards) * 15  # Assuming 15-minute episodes
            }
    return pd.DataFrame(stats).T

def show_results_analysis():
    st.title("Results Analysis")
    
    # Load all results
    results = load_all_results()
    
    if not results:
        st.warning("No results available for analysis.")
        return
        
    # Summary statistics
    st.subheader("Summary Statistics")
    stats_df = calculate_statistics(results)
    st.dataframe(stats_df.style.format({
        'Mean Reward': '{:.2f}',
        'Max Reward': '{:.2f}',
        'Min Reward': '{:.2f}',
        'Std Dev': '{:.2f}',
        'Episodes': '{:.0f}',
        'Convergence': '{:.0f}'
    }))
    
    # Reward distribution
    st.subheader("Reward Distribution")
    dist_fig = plot_reward_distribution(results)
    if dist_fig:
        st.plotly_chart(dist_fig, use_container_width=True)
        
    # Convergence comparison
    st.subheader("Convergence Comparison")
    conv_fig = plot_convergence_comparison(results)
    if conv_fig:
        st.plotly_chart(conv_fig, use_container_width=True)
        
    # Export options
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to CSV"):
            # Export statistics to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results_analysis_{timestamp}.csv'
            stats_df.to_csv(filename)
            st.success(f"Results exported to {filename}")
            
    with col2:
        if st.button("Export Raw Data"):
            # Export raw results to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'raw_results_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            st.success(f"Raw data exported to {filename}")
            
    # Additional analysis options
    st.subheader("Analysis Options")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Reward Trends", "Performance Metrics", "Convergence Analysis"]
    )
    
    if analysis_type == "Reward Trends":
        # Show reward trends analysis
        st.write("Reward Trends Analysis")
        window = st.slider("Moving Average Window", 5, 50, 10)
        
        fig = go.Figure()
        for algo, res in results.items():
            rewards = res.get('rewards', [])
            if rewards:
                ma = pd.Series(rewards).rolling(window=window).mean()
                fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma, name=algo))
                
        fig.update_layout(title=f"Reward Trends ({window}-Episode Moving Average)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Performance Metrics":
        # Show performance metrics
        st.write("Performance Metrics")
        metrics = {}
        for algo, res in results.items():
            rewards = res.get('rewards', [])
            if rewards:
                metrics[algo] = {
                    'Final Performance': np.mean(rewards[-10:]),
                    'Learning Speed': np.argmax(rewards) / len(rewards),
                    'Stability': np.std(rewards[-20:])
                }
        
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format('{:.3f}'))
        
    else:  # Convergence Analysis
        # Show convergence analysis
        st.write("Convergence Analysis")
        threshold = st.slider("Convergence Threshold (%)", 50, 95, 80)
        
        convergence_data = {}
        for algo, res in results.items():
            rewards = res.get('rewards', [])
            if rewards:
                max_reward = np.max(rewards)
                threshold_value = max_reward * (threshold / 100)
                convergence_episode = np.argmax(np.array(rewards) >= threshold_value)
                convergence_data[algo] = convergence_episode
                
        st.write(f"Episodes to reach {threshold}% of maximum reward:")
        st.table(pd.Series(convergence_data)) 