import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PLOTLY_DARK_TEMPLATE


def _df(data: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(data)
    if not frame.empty and "Timestamp" in frame.columns:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    return frame


def render_anomaly_page() -> None:
    st.title("Anomaly Detection")
    raw_df = _df(st.session_state.get("raw_data", []))
    anomaly_df = _df(st.session_state.get("anomaly_data", []))

    if raw_df.empty:
        st.info("No data found. Click Refresh Data in the sidebar to initialize the pipeline.")
        return

    # Date filter
    raw_df["_date"] = raw_df["Timestamp"].dt.date.astype(str)
    date_opts = ["All"] + sorted(raw_df["_date"].dropna().unique().tolist())
    sel_date = st.selectbox("Filter Data by Date", options=date_opts, key="anom_date")

    if sel_date != "All":
        raw_df = raw_df[raw_df["_date"] == sel_date]
        if not anomaly_df.empty:
            anomaly_df["_date"] = anomaly_df["Timestamp"].dt.date.astype(str)
            anomaly_df = anomaly_df[anomaly_df["_date"] == sel_date]

    server_options = ["All"] + sorted(raw_df["Server_ID"].dropna().unique().tolist())
    selected_server = st.selectbox("Filter by Server", options=server_options)

    if selected_server != "All":
        raw_df = raw_df[raw_df["Server_ID"] == selected_server]
        anomaly_df = anomaly_df[anomaly_df["Server_ID"] == selected_server]

    sev_counts = anomaly_df["severity_label"].value_counts() if not anomaly_df.empty else pd.Series(dtype=int)
    total = int(len(anomaly_df))
    critical = int(sev_counts.get("Critical", 0))
    high = int(sev_counts.get("High", 0))
    medium = int(sev_counts.get("Medium", 0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Anomalies", total)
    m2.markdown("<div class='custom-card' style='border-left:4px solid #f85149;'>Critical</div>", unsafe_allow_html=True)
    m2.metric("Critical Count", critical)
    m3.markdown("<div class='custom-card' style='border-left:4px solid #d29922;'>High</div>", unsafe_allow_html=True)
    m3.metric("High Count", high)
    m4.markdown("<div class='custom-card' style='border-left:4px solid #f2cc60;'>Medium</div>", unsafe_allow_html=True)
    m4.metric("Medium Count", medium)

    c1, c2 = st.columns(2)
    with c1:
        order = ["Critical", "High", "Medium"]
        sev_frame = pd.DataFrame({"severity": order, "count": [critical, high, medium]})
        fig_bar = px.bar(
            sev_frame,
            x="severity",
            y="count",
            color="severity",
            color_discrete_map={"Critical": "#f85149", "High": "#d29922", "Medium": "#f2cc60"},
            title="Severity Breakdown",
        )
        fig_bar.update_layout(**PLOTLY_DARK_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        if anomaly_df.empty:
            st.info("No anomalies to plot.")
        else:
            anomaly_df["hour_of_day"] = anomaly_df["Timestamp"].dt.hour
            by_hour = (
                anomaly_df.groupby("hour_of_day", as_index=False)
                .size()
                .rename(columns={"size": "anomaly_count"})
            )
            # Ensure all 24 hours exist in the chart
            all_hours = pd.DataFrame({"hour_of_day": range(24)})
            by_hour = pd.merge(all_hours, by_hour, on="hour_of_day", how="left").fillna(0)

            fig_hour = px.bar(
                by_hour,
                x="hour_of_day",
                y="anomaly_count",
                title="Anomalies by Hour of Day (Peak Identifier)",
                labels={"hour_of_day": "Hour of Day (24h)", "anomaly_count": "Anomaly Count"},
            )
            fig_hour.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
            fig_hour.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig_hour, use_container_width=True)

    timeline = go.Figure()
    timeline.add_trace(
        go.Scatter(
            x=raw_df["Timestamp"],
            y=raw_df["Power_Usage_Watts"],
            mode="markers",
            name="Normal",
            marker={"color": "#8b949e", "size": 5, "opacity": 0.5},
            hovertemplate="%{x}<br>Power: %{y:.2f}W<extra></extra>",
        )
    )

    color_map = {"Critical": "#f85149", "High": "#d29922", "Medium": "#f2cc60"}
    for sev in ["Critical", "High", "Medium"]:
        s = anomaly_df[anomaly_df["severity_label"] == sev]
        if s.empty:
            continue
        timeline.add_trace(
            go.Scatter(
                x=s["Timestamp"],
                y=s["Power_Usage_Watts"],
                mode="markers",
                name=sev,
                marker={"color": color_map[sev], "size": 11, "line": {"color": "#0d1117", "width": 1}},
                customdata=s[["Server_ID", "anomaly_context"]],
                hovertemplate=(
                    "%{x}<br>Server: %{customdata[0]}<br>Power: %{y:.2f}W"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )

    timeline.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
    timeline.update_layout(
        title="Anomaly Timeline Visualization",
        xaxis={"rangeslider": {"visible": True}, "title": "Time"},
        yaxis={"title": "Power Usage (Watts)"},
    )
    st.plotly_chart(timeline, use_container_width=True)

    with st.expander("How Anomaly Detection Works"):
        st.markdown(
            """
            Isolation Forest is trained per server on behavioral features (z-score, delta, energy rate, rolling std).
            DBSCAN is also run per server after scaling those features to detect density-based noise points.
            A point is flagged only when both models agree (consensus), reducing false positives.
            Feature signals used: power_z_score, power_delta, energy_rate, rolling_std_power.
            """
        )

    st.subheader("Anomaly Details")
    if anomaly_df.empty:
        st.info("No anomalies detected for the selected filter.")
        return

    show_cols = [
        "Timestamp",
        "Server_ID",
        "Power_Usage_Watts",
        "severity_label",
        "power_z_score",
        "power_delta",
        "anomaly_context",
    ]
    anomaly_df = anomaly_df.sort_values("Timestamp", ascending=False)
    st.dataframe(anomaly_df[show_cols], use_container_width=True, hide_index=True)

    csv_bytes = anomaly_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Anomalies CSV",
        data=csv_bytes,
        file_name="greenpulse_anomalies.csv",
        mime="text/csv",
    )
