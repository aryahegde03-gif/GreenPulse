import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

from config import PLOTLY_DARK_TEMPLATE
from database.mongo import raw_col

COLORS = {"S1": "#58a6ff", "S2": "#39d353", "S3": "#f2cc60"}


def _fetch_live(since: datetime) -> pd.DataFrame:
    docs = list(raw_col.find(
        {"is_live": True, "Timestamp": {"$gte": since}},
        {"_id": 0}
    ).sort("Timestamp", 1))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def _fetch_latest() -> pd.DataFrame:
    docs = list(raw_col.find(
        {"is_live": True}, {"_id": 0}
    ).sort("Timestamp", -1).limit(9))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def render_live_monitor() -> None:
    st.title("🔴 Live Emission Monitor")

    # ── Controls ──────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        refresh_sec = st.slider(
            "Update interval (s)", min_value=1, max_value=15, value=3,
            key="live_refresh_interval"
        )
    with col_b:
        window_min = st.slider(
            "Show last (minutes)", min_value=1, max_value=60, value=10,
            key="live_window_min"
        )
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        paused = st.toggle("⏸ Pause", key="live_paused", value=False)

    st.markdown(
        "<p style='color:#f85149;font-size:13px;margin:0;'>● LIVE — only this page refreshes every few seconds</p>",
        unsafe_allow_html=True,
    )
    st.caption("Run `python simulator.py` in a separate terminal to generate live data.")
    st.divider()

    # ── Fetch data ────────────────────────────────────────────────────────
    since     = datetime.utcnow() - timedelta(minutes=window_min)
    live_df   = _fetch_live(since)
    latest_df = _fetch_latest()

    # ── Status ────────────────────────────────────────────────────────────
    if latest_df.empty:
        st.warning("⚠️ No live data found. Run the simulator:\n```\npython simulator.py\n```")
        if not paused:
            time.sleep(refresh_sec)
            st.rerun()
        return

    last_ts  = latest_df["Timestamp"].max()
    secs_ago = max(0, int((datetime.utcnow() - last_ts.replace(tzinfo=None)).total_seconds()))
    st.success(f"● LIVE  |  Last reading: `{last_ts.strftime('%H:%M:%S')}`  |  {secs_ago}s ago")

    # ── Current metrics per server ─────────────────────────────────────────
    latest_per   = latest_df.sort_values("Timestamp").groupby("Server_ID").last().reset_index()
    total_power  = float(latest_per["Power_Usage_Watts"].sum())
    total_carbon = float(latest_per["carbon_kg"].sum())
    carbon_per_hour = total_carbon * 1200  # scale 3s reading → per hour

    servers = sorted(latest_per["Server_ID"].unique())
    cols = st.columns(len(servers) + 1)
    cols[0].metric("⚡ Fleet Power", f"{total_power:.1f} W", f"{total_carbon:.6f} kg CO₂")
    for i, srv in enumerate(servers):
        row = latest_per[latest_per["Server_ID"] == srv].iloc[0]
        cols[i + 1].metric(
            f"🖥️ {srv}",
            f"{row['Power_Usage_Watts']:.1f} W",
            f"{row['carbon_kg']:.5f} kg CO₂",
        )

    # ── Gauge + carbon rate card ───────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_power,
        title={"text": "Fleet Power (W)", "font": {"color": "#e6edf3", "size": 14}},
        gauge={
            "axis": {"range": [0, 600], "tickcolor": "#8b949e"},
            "bar": {"color": "#58a6ff"},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 200], "color": "#1a3a1a"},
                {"range": [200, 400], "color": "#3a320a"},
                {"range": [400, 600], "color": "#3a1010"},
            ],
            "threshold": {"line": {"color": "#f85149", "width": 3},
                          "thickness": 0.75, "value": 480},
        },
        number={"font": {"color": "#e6edf3"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0d1117", font={"color": "#e6edf3"},
        height=240, margin={"t": 50, "b": 10, "l": 20, "r": 20},
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with g2:
        st.markdown(
            f"""
            <div style="background:#161b22;border-radius:12px;padding:28px;
                        border-left:4px solid #39d353;text-align:center;margin-top:10px;">
                <h2 style="color:#39d353;margin:0;">{total_carbon:.6f}</h2>
                <p style="color:#8b949e;margin:4px 0 14px;">kg CO₂ this reading</p>
                <h3 style="color:#f2cc60;margin:0;">{carbon_per_hour:.4f}</h3>
                <p style="color:#8b949e;margin:4px 0;">kg CO₂ / hour (projected)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Rolling charts ─────────────────────────────────────────────────────
    if not live_df.empty:
        fig_power = go.Figure()
        for srv in sorted(live_df["Server_ID"].unique()):
            sdf = live_df[live_df["Server_ID"] == srv]
            fig_power.add_trace(go.Scatter(
                x=sdf["Timestamp"], y=sdf["Power_Usage_Watts"],
                mode="lines+markers", name=srv,
                line={"color": COLORS.get(srv, "#8b949e"), "width": 2},
                marker={"size": 3},
            ))
        fig_power.update_layout(
            **PLOTLY_DARK_TEMPLATE["layout"],
            title=f"Live Power — Last {window_min} min",
            height=300, margin={"t": 40, "b": 30},
        )
        st.plotly_chart(fig_power, use_container_width=True)

        fig_carbon = go.Figure()
        for srv in sorted(live_df["Server_ID"].unique()):
            sdf = live_df[live_df["Server_ID"] == srv]
            fig_carbon.add_trace(go.Scatter(
                x=sdf["Timestamp"], y=sdf["carbon_kg"],
                mode="lines", fill="tozeroy", name=srv,
                line={"color": COLORS.get(srv, "#8b949e"), "width": 2},
                opacity=0.75,
            ))
        fig_carbon.update_layout(
            **PLOTLY_DARK_TEMPLATE["layout"],
            title=f"Live CO₂ Emissions — Last {window_min} min",
            height=280, margin={"t": 40, "b": 30},
        )
        st.plotly_chart(fig_carbon, use_container_width=True)

        st.subheader("Latest Readings")
        show = live_df.sort_values("Timestamp", ascending=False).head(15)[[
            "Timestamp", "Server_ID", "Power_Usage_Watts", "energy_kwh", "carbon_kg"
        ]]
        st.dataframe(show, use_container_width=True, hide_index=True)

        t1, t2, t3 = st.columns(3)
        t1.metric("Total Readings", f"{len(live_df):,}")
        t2.metric("Total Energy (kWh)", f"{float(live_df['energy_kwh'].sum()):.4f}")
        t3.metric("Total Carbon (kg)", f"{float(live_df['carbon_kg'].sum()):.6f}")

    # ── Schedule next refresh (ONLY reruns THIS page) ─────────────────────
    if not paused:
        time.sleep(refresh_sec)
        st.rerun()   # ← reruns only live_monitor, stops when you navigate away
