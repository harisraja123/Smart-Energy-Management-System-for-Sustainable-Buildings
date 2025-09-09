import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def create_temperature_gauge(current_temp, target_temp):
    """Create a temperature gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_temp,
        delta = {'reference': target_temp},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Temperature (°C)"},
        gauge = {
            'axis': {'range': [18, 28]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [18, 20], 'color': "lightgray"},
                {'range': [20, 24], 'color': "lightgreen"},
                {'range': [24, 28], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_temp
            }
        }
    ))
    
    fig.update_layout(height=200)
    return fig

def create_energy_chart(building_id):
    """Create energy consumption chart"""
    # Simulate 24 hours of data
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24),
                         end=datetime.now(),
                         freq='H')
    
    # Simulate energy consumption with some randomness
    energy = np.random.normal(2.5, 0.5, len(hours))
    energy = np.maximum(energy, 0)  # Ensure non-negative values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=energy,
        mode='lines',
        name='Energy Consumption',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='24-Hour Energy Consumption',
        xaxis_title='Time',
        yaxis_title='Energy (kWh)',
        height=300
    )
    
    return fig

def show_environment_details():
    st.title("Environment Details")
    
    # Building selection
    building_id = st.selectbox(
        "Select Building",
        range(1, 13),
        format_func=lambda x: f"Building {x}"
    )
    
    # Current time
    st.write(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Temperature control section
    st.subheader("Temperature Control")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Temperature gauge
        current_temp = np.random.normal(22, 1)  # Simulated current temperature
        target_temp = 22.5  # Simulated target temperature
        fig = create_temperature_gauge(current_temp, target_temp)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.metric(
            "Comfort Score",
            f"{np.random.normal(85, 5):.1f}%",
            delta="1.2%"
        )
        
    with col3:
        st.metric(
            "Energy Efficiency",
            f"{np.random.normal(90, 3):.1f}%",
            delta="-0.8%"
        )
    
    # Energy consumption chart
    st.subheader("Energy Consumption")
    energy_fig = create_energy_chart(building_id)
    st.plotly_chart(energy_fig, use_container_width=True)
    
    # Environmental conditions
    st.subheader("Environmental Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Humidity",
            f"{np.random.normal(45, 5):.1f}%",
            delta="2%"
        )
        
    with col2:
        st.metric(
            "CO2 Level",
            f"{np.random.normal(600, 50):.0f} ppm",
            delta="-50 ppm"
        )
        
    with col3:
        st.metric(
            "Occupancy",
            f"{np.random.choice(['High', 'Medium', 'Low'])}"
        )
    
    # Action history
    st.subheader("Recent Actions")
    action_data = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=5, freq='15min'),
        'Action': ['Increase temp', 'Maintain', 'Decrease temp', 'Maintain', 'Increase temp'],
        'Reason': ['Low temperature', 'Optimal', 'High temperature', 'Optimal', 'Low temperature']
    })
    st.table(action_data.style.format({'Time': lambda x: x.strftime('%H:%M:%S')})) 