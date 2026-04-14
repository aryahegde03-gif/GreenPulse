import sys
sys.path.insert(0, 'greenpulse')

import pandas as pd
from pipeline.clean import clean_dataframe
from features.prediction import run_carbon_prediction, build_features

print("Reading dataset...")
df = pd.read_csv("greenpulse/dataset.csv")
print(f"Rows loaded: {len(df)}")

print("Cleaning data...")
clean_df = clean_dataframe(df)
clean_df["Timestamp"] = pd.to_datetime(clean_df["Timestamp"], errors="coerce")
print(f"Cleaned rows: {len(clean_df)}")

print("Engineered features and saving to processed_dataset.csv...")
featured_df = build_features(clean_df)
featured_df.to_csv("greenpulse/processed_dataset.csv", index=False)
print(f"Processed dataset saved with {len(featured_df)} rows and {len(featured_df.columns)} columns.")

print("Reading processed dataset for training...")
loaded_featured_df = pd.read_csv("greenpulse/processed_dataset.csv")

print("Training Random Forest Regressor on dataset...")
results = run_carbon_prediction(loaded_featured_df)

print()
print("=" * 44)
print("         MODEL TRAINING RESULTS          ")
print("=" * 44)
print(f"  Train Samples    : {results['train_samples']:,}")
print(f"  Test Samples     : {results['test_samples']:,}")
print(f"  Features Used    : {results['feature_count']}")
print("-" * 44)
print(f"  R2 Score (Train) : {results['train_r2']:.4f}")
print(f"  R2 Score (Test)  : {results['test_r2']:.4f}  <-- main metric")
print(f"  MAE              : {results['test_mae']:.6f} kg CO2")
print(f"  RMSE             : {results['test_rmse']:.6f} kg CO2")
print(f"  MAPE             : {results['test_mape']:.2f}%")
print("=" * 44)
print()
print("Top 10 Feature Importances:")
print("-" * 44)
for i, feat in enumerate(results["feature_importance"][:10], 1):
    bar = "#" * int(feat["importance"] * 300)
    print(f"  {i:2}. {feat['feature']:30s} {feat['importance']:.4f}  {bar}")
print("=" * 44)
