# Code to create a smaller sample from your existing data
import pandas as pd

# Load your large training data
df = pd.read_csv("data/processed/train_edad7fda80.csv")

# Create a representative sample (e.g., 1000 records)
sample_df = df.sample(n=1000, random_state=42)  

# Save to a new file
sample_df.to_csv("data/processed/sample_train_edad7fda80.csv", index=False)