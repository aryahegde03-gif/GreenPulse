import re

import pandas as pd

from config import EMISSION_FACTOR


def _title_case_underscore(name: str) -> str:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", str(name).strip()) if t]
    return "_".join(token.capitalize() for token in tokens)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    original_rows = len(df)

    # Rule 1: strip whitespace
    df.columns = df.columns.str.strip()
    renamed_cols = {col: _title_case_underscore(col) for col in df.columns}
    df = df.rename(columns=renamed_cols)

    canonical_lookup = {
        re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns
    }

    def pick_col(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in canonical_lookup:
                return canonical_lookup[candidate]
        return None

    timestamp_col = pick_col(["timestamp", "datetime", "time"])
    server_col = pick_col(["serverid", "server", "servername"])
    power_col = pick_col(["powerusagewatts", "powerusage", "powerwatts", "watts"])

    if not timestamp_col or not server_col or not power_col:
        raise ValueError(
            "[CLEAN] Required columns not found. Need Timestamp, Server_ID, Power_Usage_Watts."
        )

    df = df.rename(
        columns={
            timestamp_col: "Timestamp",
            server_col: "Server_ID",
            power_col: "Power_Usage_Watts",
        }
    )

    # Rule 2: parse Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    before_ts_drop = len(df)

    # Rule 3: drop invalid timestamps
    df = df.dropna(subset=["Timestamp"])
    print(f"[CLEAN] Dropped {before_ts_drop - len(df)} rows with invalid Timestamp")

    # Rule 5: cast power to float
    df["Power_Usage_Watts"] = pd.to_numeric(df["Power_Usage_Watts"], errors="coerce").astype(float)
    before_power_drop = len(df)

    # Rule 4: drop null/negative power
    df = df.dropna(subset=["Power_Usage_Watts"])
    df = df[df["Power_Usage_Watts"] >= 0]
    print(
        f"[CLEAN] Dropped {before_power_drop - len(df)} rows with null/negative Power_Usage_Watts"
    )

    # Rule 6: sort
    df = df.sort_values(["Server_ID", "Timestamp"]).reset_index(drop=True)

    # Rule 7: deduplicate timestamp+server
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Timestamp", "Server_ID"], keep="first")
    print(f"[CLEAN] Dropped {before_dedup - len(df)} duplicate Timestamp+Server_ID rows")

    # Rule 8: derived columns
    df["hours_interval"] = (
        df.groupby("Server_ID")["Timestamp"].diff().dt.total_seconds().div(3600)
    )
    df["hours_interval"] = df["hours_interval"].fillna(5 / 60)
    df.loc[df["hours_interval"] <= 0, "hours_interval"] = 5 / 60

    df["energy_kwh"] = (df["Power_Usage_Watts"] * df["hours_interval"]) / 1000
    df["carbon_kg"] = df["energy_kwh"] * EMISSION_FACTOR
    df["hour_of_day"] = df["Timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype(int)

    print(f"[CLEAN] Total dropped rows: {original_rows - len(df)}")
    print(f"[CLEAN] Final cleaned rows: {len(df)}")
    return df
