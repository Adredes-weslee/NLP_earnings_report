import sys
import os
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

# Import the actual app
try:
    from src.dashboard.app import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"Error importing dashboard: {str(e)}")
    
    # Display helpful debugging information
    st.write("### Debug Information")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Files in directory: {os.listdir('.')}")
    
    if os.path.exists("src"):
        st.write(f"Files in src: {os.listdir('src')}")
        
        if os.path.exists("src/dashboard"):
            st.write(f"Files in src/dashboard: {os.listdir('src/dashboard')}")