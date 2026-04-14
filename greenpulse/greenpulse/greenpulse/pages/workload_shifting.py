import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PLOTLY_DARK_TEMPLATE


def render_shifting_page() -> None:
    st.title("Workload Shifting Simulator")
    shift_results = st.session_state.get("shift_results", {}) or {}

    if not shift_results:
        st.info("No shift simulation results found. Click Refresh Data in the sidebar.")
        return

    total_actual = float(shift_results.get("total_actual_carbon_kg", 0.0))
    total_shifted = float(shift_results.get("total_shifted_carbon_kg", 0.0))
    total_saved = float(shift_results.get("total_carbon_saved_kg", 0.0))
    pct_saved = float(shift_results.get("fleet_percent_savings", 0.0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Actual Carbon (kg)", f"{total_actual:.2f}")
    m2.metric("Total After Shifting (kg)", f"{total_shifted:.2f}")
    m3.metric("Carbon Saved (kg)", f"{total_saved:.2f}", f"+{total_saved:.2f}")
    m4.metric("Savings Percentage", f"{pct_saved:.2f}%", f"+{pct_saved:.2f}%")

    intensity_dict = shift_results.get("hourly_intensity_profile", {})
    green_hours = shift_results.get("green_hours", [])
    red_hours = shift_results.get("red_hours", [])

    intensity_df = pd.DataFrame(
        {
            "hour": [int(h) for h in intensity_dict.keys()],
            "intensity": [float(v) for v in intensity_dict.values()],
        }
    ).sort_values("hour")

    fig_intensity = go.Figure()
    fig_intensity.add_trace(
        go.Scatter(
            x=intensity_df["hour"],
            y=intensity_df["intensity"],
            mode="lines+markers",
            fill="tozeroy",
            name="Carbon Intensity",
            line={"color": "#58a6ff", "width": 3},
        )
    )
    for h in green_hours:
        fig_intensity.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor="#39d353", opacity=0.15, line_width=0)
    for h in red_hours:
        fig_intensity.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor="#f85149", opacity=0.12, line_width=0)
    fig_intensity.add_annotation(x=green_hours[0] if green_hours else 0, y=float(intensity_df["intensity"].min()), text="Green Window", showarrow=False)
    fig_intensity.add_annotation(x=red_hours[0] if red_hours else 0, y=float(intensity_df["intensity"].max()), text="Peak Hours", showarrow=False)
    fig_intensity.update_layout(title="Carbon Intensity Profile", **PLOTLY_DARK_TEMPLATE["layout"])
    st.plotly_chart(fig_intensity, use_container_width=True)

    gcol, rcol = st.columns(2)
    gcol.success(f"Leaf Green Hours: {green_hours}")
    rcol.error(f"Fire Peak Red Hours: {red_hours}")

    fcol, icol = st.columns(2)
    flexible = shift_results.get("flexible_servers", [])
    inflexible = shift_results.get("inflexible_servers", [])
    fcol.markdown(
        f"<div class='custom-card'><h4>Flexible Servers</h4><p>{', '.join(flexible) if flexible else 'None'}</p></div>",
        unsafe_allow_html=True,
    )
    icol.markdown(
        f"<div class='custom-card'><h4>Inflexible Servers</h4><p>{', '.join(inflexible) if inflexible else 'None'}</p></div>",
        unsafe_allow_html=True,
    )

    per_server = pd.DataFrame(shift_results.get("per_server_results", []))
    if not per_server.empty:
        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Bar(
                x=per_server["server_id"],
                y=per_server["actual_carbon_kg"],
                name="Actual",
                marker_color="#8b949e",
            )
        )
        fig_compare.add_trace(
            go.Bar(
                x=per_server["server_id"],
                y=per_server["shifted_carbon_kg"],
                name="Shifted",
                marker_color="#39d353",
            )
        )
        for _, row in per_server.iterrows():
            fig_compare.add_annotation(
                x=row["server_id"],
                y=max(row["actual_carbon_kg"], row["shifted_carbon_kg"]),
                text=f"-{row['carbon_saved_kg']:.2f}",
                showarrow=False,
                yshift=10,
            )
        fig_compare.update_layout(
            title="Actual vs Shifted Comparison",
            barmode="group",
            **PLOTLY_DARK_TEMPLATE["layout"],
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Recommendations")
    for item in shift_results.get("per_server_results", []):
        border = "#39d353" if item["flexibility"] == "flexible" else "#8b949e"
        st.markdown(
            (
                f"<div class='custom-card' style='border-left: 4px solid {border};'>"
                f"<h4>{item['server_id']} ({item['flexibility']})</h4>"
                f"<p>{item['recommendation']}</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Shifting Methodology"):
        st.markdown(
            """
            Flexible workloads are inferred per server using coefficient of variation and peak-hour behavior.
            Carbon intensity is derived from hourly fleet power usage and normalized to a realistic proxy range.
            Flexible workloads observed in red hours are virtually shifted to average green-hour intensity.
            This follows the same simulation spirit as carbon-intelligent scheduling systems.
            """
        )
