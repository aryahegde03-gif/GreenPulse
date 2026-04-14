import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from database.mongo import anomaly_col


def _safe_server_mean(series: pd.Series, default: float = 0.0) -> float:
    val = float(series.mean()) if not series.empty else default
    if np.isnan(val):
        return default
    return val


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("[ANOMALY] Empty dataframe; skipping anomaly detection")
        try:
            anomaly_col.delete_many({})
        except Exception as exc:
            print(f"[ANOMALY] Failed to clear anomaly collection: {exc}")
        return df

    work_df = df.copy()
    work_df["Timestamp"] = pd.to_datetime(work_df["Timestamp"], errors="coerce")
    work_df = work_df.sort_values(["Server_ID", "Timestamp"]).reset_index(drop=True)

    all_parts = []
    feature_cols = ["power_z_score", "power_delta", "energy_rate", "rolling_std_power"]

    for server_id, server_df in work_df.groupby("Server_ID", group_keys=False):
        part = server_df.copy()

        part["rolling_mean_power"] = part["Power_Usage_Watts"].rolling(window=12, min_periods=1).mean()
        part["rolling_std_power"] = part["Power_Usage_Watts"].rolling(window=12, min_periods=1).std()
        part["power_z_score"] = (
            (part["Power_Usage_Watts"] - part["rolling_mean_power"])
            / (part["rolling_std_power"] + 1e-9)
        )
        part["power_delta"] = part["Power_Usage_Watts"].diff()
        part["energy_rate"] = part["energy_kwh"] / (part["hours_interval"] + 1e-9)

        for col in ["rolling_mean_power", "rolling_std_power", "power_z_score", "power_delta", "energy_rate"]:
            part[col] = part[col].fillna(_safe_server_mean(part[col]))

        if len(part) >= 5:
            # More sensitive: 2% contamination
            iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
            part["iso_label"] = iso.fit_predict(part[feature_cols])

            scaler = StandardScaler()
            scaled = scaler.fit_transform(part[feature_cols])
            # Much more sensitive epsilon for standardized features
            db = DBSCAN(eps=0.3, min_samples=3, metric="euclidean")
            part["dbscan_label"] = db.fit_predict(scaled)
        else:
            part["iso_label"] = 1
            part["dbscan_label"] = 0

        # Change to Union logic: if EITHER model is confident
        part["is_anomaly"] = (part["iso_label"] == -1) | (part["dbscan_label"] == -1)

        part["severity"] = (
            np.abs(part["power_z_score"])
            * (part["power_delta"] / (part["rolling_mean_power"] + 1e-9))
        )
        part.loc[~part["is_anomaly"], "severity"] = np.nan
        part["severity_label"] = np.where(
            part["is_anomaly"] & (part["severity"] > 2),
            "Critical",
            np.where(part["is_anomaly"] & (part["severity"] > 1), "High", np.where(part["is_anomaly"], "Medium", "")),
        )

        part["anomaly_context"] = ""
        anomaly_mask = part["is_anomaly"]
        part.loc[anomaly_mask, "anomaly_context"] = part.loc[anomaly_mask].apply(
            lambda row: (
                f"Server {row['Server_ID']} drew {row['Power_Usage_Watts']}W at {row['Timestamp']} "
                f"- {row['power_delta']:+.1f}W vs previous reading, "
                f"{abs(row['power_z_score']):.1f} std deviations from its 1-hour baseline "
                f"(baseline mean: {row['rolling_mean_power']:.1f}W)"
            ),
            axis=1,
        )

        print(
            f"[ANOMALY] Isolation Forest + DBSCAN complete for server {server_id}: "
            f"{int(part['is_anomaly'].sum())} anomalies"
        )
        all_parts.append(part)

    result_df = pd.concat(all_parts).sort_values(["Server_ID", "Timestamp"]).reset_index(drop=True)

    anomalies_df = result_df[result_df["is_anomaly"]].copy()
    anomalies_records = anomalies_df.to_dict("records")
    for record in anomalies_records:
        if isinstance(record.get("Timestamp"), pd.Timestamp):
            record["Timestamp"] = record["Timestamp"].to_pydatetime()

    try:
        anomaly_col.delete_many({})
        if anomalies_records:
            anomaly_col.insert_many(anomalies_records)
    except Exception as exc:
        print(f"[ANOMALY] Failed to write anomalies to MongoDB: {exc}")
    print(f"[ANOMALY] Stored {len(anomalies_records)} consensus anomalies")

    return result_df
