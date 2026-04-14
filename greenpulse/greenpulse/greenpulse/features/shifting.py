import numpy as np
import pandas as pd

from database.mongo import shift_col


def run_shift_simulation(df: pd.DataFrame) -> dict:
    if df.empty:
        print("[SHIFT] Empty dataframe; skipping simulation")
        try:
            shift_col.delete_many({})
        except Exception as exc:
            print(f"[SHIFT] Failed to clear shift collection: {exc}")
        empty = {
            "total_actual_carbon_kg": 0.0,
            "total_shifted_carbon_kg": 0.0,
            "total_carbon_saved_kg": 0.0,
            "fleet_percent_savings": 0.0,
            "green_hours": [],
            "red_hours": [],
            "flexible_servers": [],
            "inflexible_servers": [],
            "hourly_intensity_profile": {},
            "per_server_results": [],
        }
        try:
            shift_col.insert_one(empty)
        except Exception as exc:
            print(f"[SHIFT] Failed to write empty simulation: {exc}")
        return empty

    sim_df = df.copy()
    sim_df["Timestamp"] = pd.to_datetime(sim_df["Timestamp"], errors="coerce")

    # Step 1: synthetic hourly intensity profile
    hourly_power = sim_df.groupby("hour_of_day", as_index=False)["Power_Usage_Watts"].mean()
    min_power = float(hourly_power["Power_Usage_Watts"].min())
    max_power = float(hourly_power["Power_Usage_Watts"].max())
    denom = max(max_power - min_power, 1e-9)
    hourly_power["normalized"] = (hourly_power["Power_Usage_Watts"] - min_power) / denom
    hourly_power["intensity"] = 0.6 + (hourly_power["normalized"] * 0.4)

    hourly_intensity = {
        int(row["hour_of_day"]): float(row["intensity"])
        for _, row in hourly_power.iterrows()
    }

    # Step 2: green and red windows
    intensity_values = np.array(list(hourly_intensity.values()))
    low_cut = float(np.percentile(intensity_values, 25))
    high_cut = float(np.percentile(intensity_values, 75))
    green_hours = sorted([h for h, v in hourly_intensity.items() if v <= low_cut])
    red_hours = sorted([h for h, v in hourly_intensity.items() if v >= high_cut])
    avg_green_intensity = float(np.mean([hourly_intensity[h] for h in green_hours])) if green_hours else 0.6
    print(f"[SHIFT] Green windows identified: {green_hours}")

    # Step 3: classify flexibility
    flexibility_map = {}
    for server_id, sdf in sim_df.groupby("Server_ID"):
        mean_power = float(sdf["Power_Usage_Watts"].mean())
        std_power = float(sdf["Power_Usage_Watts"].std(ddof=0))
        cv = std_power / (mean_power + 1e-9)

        max_power = float(sdf["Power_Usage_Watts"].max())
        peak_rows = sdf[sdf["Power_Usage_Watts"] == max_power]
        if len(peak_rows) == 0:
            red_peak_ratio = 0.0
        else:
            red_peak_ratio = float(peak_rows["hour_of_day"].isin(red_hours).mean())

        is_flexible = (cv < 0.2) and (red_peak_ratio > 0.5)
        flexibility_map[server_id] = "flexible" if is_flexible else "inflexible"

    # Step 4 and 5
    sim_df["actual_carbon"] = sim_df["carbon_kg"]

    def shifted_value(row: pd.Series) -> float:
        if (
            flexibility_map.get(row["Server_ID"]) == "flexible"
            and int(row["hour_of_day"]) in red_hours
        ):
            return float(row["energy_kwh"] * avg_green_intensity)
        return float(row["actual_carbon"])

    sim_df["shifted_carbon"] = sim_df.apply(shifted_value, axis=1)

    # Step 6
    per_server_results = []
    for server_id, sdf in sim_df.groupby("Server_ID"):
        actual = float(sdf["actual_carbon"].sum())
        shifted = float(sdf["shifted_carbon"].sum())
        saved = actual - shifted
        pct = (saved / actual * 100) if actual > 0 else 0.0
        red_hour_readings = int(sdf["hour_of_day"].isin(red_hours).sum())
        flexibility = flexibility_map[server_id]

        if flexibility == "flexible":
            recommendation = (
                f"Defer batch workloads from peak hours {red_hours} to green windows {green_hours} "
                f"- projected saving: {saved:.3f}kg CO2 ({pct:.2f}% reduction)"
            )
        else:
            recommendation = (
                "Workload appears interactive or time-sensitive - temporal shifting not recommended. "
                "Focus on efficiency optimization."
            )

        per_server_results.append(
            {
                "server_id": server_id,
                "flexibility": flexibility,
                "actual_carbon_kg": actual,
                "shifted_carbon_kg": shifted,
                "carbon_saved_kg": saved,
                "percent_savings": pct,
                "red_hour_readings": red_hour_readings,
                "recommendation": recommendation,
            }
        )

    # Step 7
    total_actual = float(sim_df["actual_carbon"].sum())
    total_shifted = float(sim_df["shifted_carbon"].sum())
    total_saved = total_actual - total_shifted
    fleet_pct = (total_saved / total_actual * 100) if total_actual > 0 else 0.0

    flexible_servers = sorted([s for s, f in flexibility_map.items() if f == "flexible"])
    inflexible_servers = sorted([s for s, f in flexibility_map.items() if f == "inflexible"])

    summary = {
        "total_actual_carbon_kg": total_actual,
        "total_shifted_carbon_kg": total_shifted,
        "total_carbon_saved_kg": total_saved,
        "fleet_percent_savings": fleet_pct,
        "green_hours": green_hours,
        "red_hours": red_hours,
        "flexible_servers": flexible_servers,
        "inflexible_servers": inflexible_servers,
        "hourly_intensity_profile": {str(k): float(v) for k, v in hourly_intensity.items()},
        "per_server_results": per_server_results,
    }

    # Step 8
    try:
        shift_col.delete_many({})
        shift_col.insert_one(summary)
    except Exception as exc:
        print(f"[SHIFT] Failed to store simulation results: {exc}")
    print(f"[SHIFT] Simulation stored. Fleet saving: {total_saved:.3f}kg CO2")

    return summary
