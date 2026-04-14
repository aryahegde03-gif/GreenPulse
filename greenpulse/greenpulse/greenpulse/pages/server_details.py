import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PLOTLY_DARK_TEMPLATE


def _frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if not df.empty and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def render_server_details() -> None:
    st.title("Server Details")
    raw_df = _frame(st.session_state.get("raw_data", []))
    anomaly_df = _frame(st.session_state.get("anomaly_data", []))
    shift_results = st.session_state.get("shift_results", {}) or {}

    if raw_df.empty:
        st.info("No data found. Click Refresh Data in the sidebar to initialize the pipeline.")
        return

    # Date filter
    raw_df["_date"] = raw_df["Timestamp"].dt.date.astype(str)
    date_opts = ["All"] + sorted(raw_df["_date"].dropna().unique().tolist())
    sel_date = st.selectbox("Filter Data by Date", options=date_opts, key="server_date")

    if sel_date != "All":
        raw_df = raw_df[raw_df["_date"] == sel_date]
        if not anomaly_df.empty:
            anomaly_df["_date"] = anomaly_df["Timestamp"].dt.date.astype(str)
            anomaly_df = anomaly_df[anomaly_df["_date"] == sel_date]

    servers = sorted(raw_df["Server_ID"].dropna().unique().tolist())
    if not servers:
        st.warning("No servers found for this date.")
        return
    default_server = st.session_state.get("selected_server", servers[0])
    selected_server = st.selectbox("Select Server", options=servers, index=servers.index(default_server) if default_server in servers else 0)
    st.session_state.selected_server = selected_server

    sdf = raw_df[raw_df["Server_ID"] == selected_server].sort_values("Timestamp")
    sadf = anomaly_df[anomaly_df["Server_ID"] == selected_server].sort_values("Timestamp") if not anomaly_df.empty else anomaly_df

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", int(len(sdf)))
    m2.metric("Total Energy (kWh)", f"{float(sdf['energy_kwh'].sum()):.3f}")
    m3.metric("Total Carbon (kg)", f"{float(sdf['carbon_kg'].sum()):.3f}")
    m4.metric("Average Power (W)", f"{float(sdf['Power_Usage_Watts'].mean()):.2f}")
    m5.metric("Anomaly Count", int(len(sadf)))

    show_roll = st.checkbox("Show rolling average", value=True)
    timeline = go.Figure()
    timeline.add_trace(
        go.Scatter(
            x=sdf["Timestamp"],
            y=sdf["Power_Usage_Watts"],
            mode="lines",
            name=f"{selected_server} Power",
            line={"color": "#58a6ff", "width": 2},
        )
    )
    if show_roll:
        rolling = sdf["Power_Usage_Watts"].rolling(12, min_periods=1).mean()
        timeline.add_trace(
            go.Scatter(
                x=sdf["Timestamp"],
                y=rolling,
                mode="lines",
                name="1h Rolling Avg",
                line={"color": "#39d353", "dash": "dash"},
            )
        )
    if not sadf.empty:
        timeline.add_trace(
            go.Scatter(
                x=sadf["Timestamp"],
                y=sadf["Power_Usage_Watts"],
                mode="markers",
                name="Anomalies",
                marker={"color": "#f85149", "size": 10, "symbol": "circle"},
            )
        )
    timeline.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
    timeline.update_layout(
        title="Power Usage Timeline",
        xaxis={"rangeslider": {"visible": True}, "title": "Time"},
        yaxis={"title": "Power Usage (Watts)"},
    )
    st.plotly_chart(timeline, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=sdf["Power_Usage_Watts"],
                nbinsx=20,
                marker_color="#58a6ff",
                opacity=0.75,
                name="Power Histogram",
            )
        )
        mean_val = float(sdf["Power_Usage_Watts"].mean())
        std_val = float(sdf["Power_Usage_Watts"].std(ddof=0) + 1e-9)
        x_vals = np.linspace(float(sdf["Power_Usage_Watts"].min()), float(sdf["Power_Usage_Watts"].max()), 200)
        normal = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_val) / std_val) ** 2)
        normal_scaled = normal * len(sdf) * (x_vals[1] - x_vals[0])
        fig_hist.add_trace(go.Scatter(x=x_vals, y=normal_scaled, mode="lines", name="Normal Overlay", line={"color": "#39d353"}))
        fig_hist.update_layout(
            title="Power Usage Distribution", 
            **PLOTLY_DARK_TEMPLATE["layout"],
            xaxis_title="Power Usage (Watts)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        others = raw_df[raw_df["Server_ID"] != selected_server]
        fig_box = go.Figure()
        fig_box.add_trace(
            go.Box(
                y=others["Power_Usage_Watts"],
                name="Fleet (Others)",
                marker_color="#8b949e",
                opacity=0.35,
            )
        )
        fig_box.add_trace(
            go.Box(
                y=sdf["Power_Usage_Watts"],
                name=selected_server,
                marker_color="#58a6ff",
            )
        )
        fig_box.update_layout(
            title="Box Plot Comparison", 
            **PLOTLY_DARK_TEMPLATE["layout"],
            xaxis_title="Server Group",
            yaxis_title="Power Usage (Watts)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    hourly_server = sdf.groupby("hour_of_day", as_index=False)["Power_Usage_Watts"].mean()
    hourly_fleet = raw_df.groupby("hour_of_day", as_index=False)["Power_Usage_Watts"].mean()
    fig_hour = go.Figure()
    fig_hour.add_trace(
        go.Scatter(
            x=hourly_server["hour_of_day"],
            y=hourly_server["Power_Usage_Watts"],
            mode="lines+markers",
            name=f"{selected_server} Avg",
            line={"color": "#58a6ff"},
        )
    )
    fig_hour.add_trace(
        go.Scatter(
            x=hourly_fleet["hour_of_day"],
            y=hourly_fleet["Power_Usage_Watts"],
            mode="lines+markers",
            name="Fleet Avg",
            line={"color": "#39d353", "dash": "dash"},
        )
    )
    fig_hour.update_layout(
        title="Hourly Pattern Analysis", 
        **PLOTLY_DARK_TEMPLATE["layout"],
        xaxis_title="Hour of Day",
        yaxis_title="Average Power Usage (Watts)"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Server Anomaly Table")
    if sadf.empty:
        st.info("No anomalies detected for this server.")
    else:
        cols = [
            "Timestamp",
            "Server_ID",
            "Power_Usage_Watts",
            "severity_label",
            "power_z_score",
            "power_delta",
            "anomaly_context",
        ]
        st.dataframe(sadf[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download Server Anomalies",
            data=sadf[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_server}_anomalies.csv",
            mime="text/csv",
        )

    st.subheader("Server Shifting Recommendation")
    rec = None
    for item in shift_results.get("per_server_results", []):
        if item.get("server_id") == selected_server:
            rec = item
            break
    if rec:
        color = "#39d353" if rec["flexibility"] == "flexible" else "#8b949e"
        st.markdown(
            (
                f"<div class='custom-card' style='border-left: 4px solid {color};'>"
                f"<h4>Status: {rec['flexibility']}</h4>"
                f"<p>{rec['recommendation']}</p>"
                f"<p>Potential saving: {rec['carbon_saved_kg']:.3f} kg CO2 ({rec['percent_savings']:.2f}%)</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info("No recommendation available yet.")
