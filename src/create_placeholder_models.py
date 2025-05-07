"""
Create placeholder models for the Streamlit dashboard.
This script generates minimal placeholder models that can be used by the 
Streamlit dashboard when the full NLP pipeline hasn't been run yet.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import placeholder model creation function
from utils import create_placeholder_models

if __name__ == "__main__":
    print("Creating placeholder models for the Streamlit dashboard...")
    success = create_placeholder_models()
    if success:
        print("Placeholder models created successfully!")
    else:
        print("Failed to create placeholder models.")