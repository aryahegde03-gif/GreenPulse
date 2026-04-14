import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-series features for supervised carbon prediction."""
    feat = df.copy()
    feat = feat.sort_values(["Server_ID", "Timestamp"]).reset_index(drop=True)

    # --- Time-based features ---
    feat["hour_of_day"] = feat["Timestamp"].dt.hour
    feat["day_of_week"] = feat["Timestamp"].dt.dayofweek
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour_of_day"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour_of_day"] / 24)

    # --- Lag features (per server) ---
    for lag in [1, 2, 3, 6, 12]:
        feat[f"power_lag_{lag}"] = feat.groupby("Server_ID")["Power_Usage_Watts"].shift(lag)
        feat[f"carbon_lag_{lag}"] = feat.groupby("Server_ID")["carbon_kg"].shift(lag)

    # --- Rolling window features (per server) ---
    for window in [6, 12, 24]:
        feat[f"power_roll_mean_{window}"] = (
            feat.groupby("Server_ID")["power_lag_1"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        feat[f"power_roll_std_{window}"] = (
            feat.groupby("Server_ID")["power_lag_1"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # --- Rate of change ---
    feat["power_diff"] = feat["power_lag_1"] - feat["power_lag_2"]
    feat["power_pct_change"] = feat["power_diff"] / (feat["power_lag_2"] + 1e-9)

    # --- Cumulative energy per server per day ---
    feat["date"] = feat["Timestamp"].dt.date

    # Fill NaNs from lag/rolling with column means
    for col in feat.columns:
        if feat[col].dtype in [np.float64, np.float32, float]:
            feat[col] = feat[col].fillna(feat[col].mean())

    # --- Server Identity ---
    server_dummies = pd.get_dummies(feat["Server_ID"], prefix="server").astype(int)
    feat = pd.concat([feat, server_dummies], axis=1)

    return feat


def run_carbon_prediction(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest Regressor to predict carbon_kg from engineered features.
    Returns a dict with model metrics, feature importances, and predictions.
    """
    if df.empty or "carbon_kg" not in df.columns:
        print("[PREDICT] Empty dataframe or missing carbon_kg; skipping prediction")
        return {}

    work_df = df.copy()
    work_df["Timestamp"] = pd.to_datetime(work_df["Timestamp"], errors="coerce")
    work_df = work_df.dropna(subset=["Timestamp", "carbon_kg"])

    # Build features if not already built
    if "hour_of_day" not in work_df.columns or "power_lag_1" not in work_df.columns:
        feat_df = build_features(work_df)
    else:
        feat_df = work_df.copy()

    # Select feature columns (exclude target, IDs, timestamps, raw derived cols)
    exclude_cols = {
        "Timestamp", "Server_ID", "carbon_kg", "date", "_id",
        "Power_Usage_Watts", "energy_kwh", "hours_interval", "is_anomaly", "iso_label",
        "dbscan_label", "severity", "severity_label", "anomaly_context",
        "rolling_mean_power", "rolling_std_power", "power_z_score",
        "actual_carbon", "shifted_carbon",
    }
    feature_cols = [
        c for c in feat_df.columns
        if c not in exclude_cols and feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]
    ]

    if len(feature_cols) < 3:
        print(f"[PREDICT] Not enough features ({len(feature_cols)}); skipping")
        return {}

    # Sort chronologically so sequential time split includes all servers
    feat_df = feat_df.sort_values("Timestamp")

    X = feat_df[feature_cols].copy()
    # Smooth the target to predict the underlying trend instead of instant noise
    y = feat_df.groupby("Server_ID")["carbon_kg"].transform(lambda x: x.rolling(window=5, min_periods=1, center=True).mean()).copy()

    # Drop any remaining NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 50:
        print(f"[PREDICT] Not enough samples ({len(X)}); skipping")
        return {}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Train Gradient Boosting Regressor
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_mape = float(np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-9))) * 100)

    # Feature importance
    importances = model.feature_importances_
    feat_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build prediction DataFrame for visualization
    test_timestamps = feat_df.loc[X_test.index, "Timestamp"].values
    test_servers = feat_df.loc[X_test.index, "Server_ID"].values
    pred_df = pd.DataFrame({
        "Timestamp": test_timestamps,
        "Server_ID": test_servers,
        "actual_carbon_kg": y_test.values,
        "predicted_carbon_kg": y_pred_test,
    })
    pred_df["error"] = pred_df["actual_carbon_kg"] - pred_df["predicted_carbon_kg"]
    pred_df["abs_error"] = np.abs(pred_df["error"])

    results = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": len(feature_cols),
        "feature_importance": [
            {"feature": name, "importance": float(imp)}
            for name, imp in feat_importance[:15]
        ],
        "predictions": pred_df.to_dict("records"),
        "feature_columns": feature_cols,
    }

    print(f"[PREDICT] Model trained — R²: {test_r2:.4f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    return results
