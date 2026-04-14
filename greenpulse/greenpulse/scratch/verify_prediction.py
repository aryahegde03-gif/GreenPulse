import pandas as pd
import sys
import os

# Add the greenpulse directory to the path so we can import our modules
sys.path.append(os.path.abspath('greenpulse'))

from pipeline.clean import clean_dataframe
from features.prediction import run_carbon_prediction

try:
    print("Reading scratch/test_input.csv...")
    df = pd.read_csv('scratch/test_input.csv')
    
    print("Cleaning dataframe...")
    cleaned_df = clean_dataframe(df)
    
    print("Running carbon prediction...")
    results = run_carbon_prediction(cleaned_df)
    
    if results:
        print("Success! R2 Score:", results.get('test_r2'))
        print("Predictions count:", len(results.get('predictions', [])))
    else:
        print("Model returned empty results (expected if too few rows).")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
