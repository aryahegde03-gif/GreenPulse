import pandas as pd
import plotly.express as px
import streamlit as st

from config import PLOTLY_DARK_TEMPLATE


def _to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if not df.empty and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def render_dashboard() -> None:
    st.title("GreenPulse Command Center")
    raw_df = _to_df(st.session_state.get("raw_data", []))
    anomaly_df = _to_df(st.session_state.get("anomaly_data", []))
    shift_results = st.session_state.get("shift_results", {}) or {}

    if raw_df.empty:
        st.info("No data found. Click Refresh Data in the sidebar to initialize the pipeline.")
        return

    # Date filter
    raw_df["_date"] = raw_df["Timestamp"].dt.date.astype(str)
    date_opts = ["All"] + sorted(raw_df["_date"].dropna().unique().tolist())
    sel_date = st.selectbox("Filter Data by Date", options=date_opts, key="dash_date")
    if sel_date != "All":
        raw_df = raw_df[raw_df["_date"] == sel_date]
        if not anomaly_df.empty:
            anomaly_df["_date"] = anomaly_df["Timestamp"].dt.date.astype(str)
            anomaly_df = anomaly_df[anomaly_df["_date"] == sel_date]

    total_carbon = float(raw_df["carbon_kg"].sum())
    total_anomalies = int(len(anomaly_df))
    saved = float(shift_results.get("total_carbon_saved_kg", 0.0))
    fleet_savings_pct = float(shift_results.get("fleet_percent_savings", 0.0))

    mid_idx = max(len(raw_df) // 2, 1)
    prev_period = float(raw_df.iloc[:mid_idx]["carbon_kg"].sum())
    curr_period = float(raw_df.iloc[mid_idx:]["carbon_kg"].sum())
    delta_carbon = curr_period - prev_period

    sev_counts = anomaly_df["severity_label"].value_counts().to_dict() if not anomaly_df.empty else {}
    sev_delta = f"C:{sev_counts.get('Critical', 0)} H:{sev_counts.get('High', 0)} M:{sev_counts.get('Medium', 0)}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Carbon Emitted (kg CO2)", f"{total_carbon:.2f}", f"{delta_carbon:+.2f} vs prev")
    c2.metric("Total Anomalies Detected", f"{total_anomalies}", sev_delta)
    c3.metric("Carbon Saved via Shifting (kg)", f"{saved:.2f}")
    c4.metric("Fleet Savings Percentage", f"{fleet_savings_pct:.2f}%")

    left, right = st.columns(2)
    with left:
        hourly = raw_df.copy()
        hourly["hour_ts"] = hourly["Timestamp"].dt.floor("H")
        trend = (
            hourly.groupby(["hour_ts", "Server_ID"], as_index=False)["carbon_kg"].sum()
            .sort_values("hour_ts")
        )
        fig_trend = px.line(
            trend,
            x="hour_ts",
            y="carbon_kg",
            color="Server_ID",
            title="Carbon Emissions Over Time",
            markers=True,
        )
        fig_trend.update_layout(
            **PLOTLY_DARK_TEMPLATE["layout"],
            xaxis_title="Time",
            yaxis_title="Carbon Emissions (kg CO₂)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with right:
        fig_box = px.box(
            raw_df,
            x="Server_ID",
            y="Power_Usage_Watts",
            color="Server_ID",
            points="outliers",
            title="Server Power Distribution",
        )
        fig_box.update_layout(
            **PLOTLY_DARK_TEMPLATE["layout"], 
            showlegend=False,
            xaxis_title="Server ID",
            yaxis_title="Power Usage (Watts)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        heat = (
            raw_df.groupby(["day_of_week", "hour_of_day"], as_index=False)["carbon_kg"]
            .mean()
            .pivot(index="day_of_week", columns="hour_of_day", values="carbon_kg")
            .fillna(0)
        )
        fig_heat = px.imshow(
            heat,
            aspect="auto",
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Avg Carbon (kg)"},
            title="Hourly Carbon Intensity Heatmap",
        )
        fig_heat.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_heat, use_container_width=True)

    with right2:
        energy = raw_df.groupby("Server_ID", as_index=False)["energy_kwh"].sum()
        fig_donut = px.pie(
            energy,
            names="Server_ID",
            values="energy_kwh",
            hole=0.5,
            title="Energy Consumption by Server",
        )
        fig_donut.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("Recent Activity")
    recent = raw_df.sort_values("Timestamp", ascending=False).head(20)[
        ["Timestamp", "Server_ID", "Power_Usage_Watts", "energy_kwh", "carbon_kg"]
    ]
    styled = recent.style.background_gradient(subset=["carbon_kg"], cmap="YlOrRd")
    st.dataframe(styled, use_container_width=True, hide_index=True)
