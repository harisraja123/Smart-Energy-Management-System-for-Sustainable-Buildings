import streamlit as st
import os
import sys
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views.training import show_training_progress
from views.testing import show_testing_progress
from views.environment import show_environment_details
from views.comparison import show_comparison_results

# Page configuration
st.set_page_config(
    page_title="Smart Energy RL Dashboard",
    page_icon="🏢",
    layout="wide"
)

# Hide default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="stDecoration"] {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    </style>
    """, unsafe_allow_html=True)

def is_test_mode():
    """Check if we're in test mode by checking active sessions."""
    try:
        # Check testing file
        test_file = os.path.join('dashboard', 'data', 'current_testing.json')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_data = json.load(f)
                if test_data.get('active', False):
                    return True
        
        # Check training file
        train_file = os.path.join('dashboard', 'data', 'current_training.json')
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_data = json.load(f)
                if train_data.get('active', False):
                    return False
    except:
        pass
    return False

def main():
    # Sidebar navigation
    #st.sidebar.title("Navigation")
    
    # Add logo or project title
    st.sidebar.markdown("# Smart Energy RL")
    st.sidebar.markdown("---")
    
    # Navigation selection - default to Testing Monitor if in test mode
    default_page = "Testing Monitor" if is_test_mode() else "Training Monitor"
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["Training Monitor", "Testing Monitor", "Environment Details", "Results Comparison"],
        index=["Training Monitor", "Testing Monitor", "Environment Details", "Results Comparison"].index(default_page)
    )
    
    # Display selected page
    if page == "Training Monitor":
        show_training_progress()
    elif page == "Testing Monitor":
        show_testing_progress()
    elif page == "Environment Details":
        show_environment_details()
    else:
        show_comparison_results()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides real-time monitoring and "
        "analysis tools for the Smart Energy RL project."
    )

if __name__ == "__main__":
    main() 