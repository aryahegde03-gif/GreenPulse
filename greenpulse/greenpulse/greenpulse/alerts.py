"""
alerts.py — GreenPulse Alert System
=====================================
Evaluates live session data against configurable thresholds
and renders Streamlit popup-style alerts (toast + expander banners).

Thresholds (all configurable in config.py):
  ALERT_POWER_CRITICAL_W   — single reading above this → Critical popup
  ALERT_POWER_WARNING_W    — single reading above this → Warning popup
  ALERT_CARBON_CRITICAL_KG — total carbon above this   → Critical popup
  ALERT_ANOMALY_CRITICAL   — anomaly count above this  → Critical popup
  ALERT_ANOMALY_WARNING     — anomaly count above this  → Warning popup

Usage (call from any page after session data is loaded):
    from alerts import evaluate_and_show_alerts
    evaluate_and_show_alerts()
"""

import pandas as pd
import streamlit as st

from config import (
    ALERT_ANOMALY_CRITICAL,
    ALERT_ANOMALY_WARNING,
    ALERT_CARBON_CRITICAL_KG,
    ALERT_POWER_CRITICAL_W,
    ALERT_POWER_WARNING_W,
)


# ── Internal helpers ──────────────────────────────────────────

def _raw_df() -> pd.DataFrame:
    data = st.session_state.get("raw_data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def _anomaly_df() -> pd.DataFrame:
    data = st.session_state.get("anomaly_data", [])
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _render_alert(level: str, title: str, body: str, detail: str = "") -> None:
    """
    Render a styled alert banner inside the Streamlit page.
    level: "critical" | "warning" | "info"
    """
    colors = {
        "critical": ("#f85149", "#3d1a1a"),
        "warning":  ("#d29922", "#3d2e0a"),
        "info":     ("#58a6ff", "#0d2137"),
    }
    icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    border_color, bg_color = colors.get(level, colors["info"])
    icon = icons.get(level, "ℹ️")

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-left: 5px solid {border_color};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        ">
            <div style="font-size:15px; font-weight:700; color:{border_color};">
                {icon} {title}
            </div>
            <div style="font-size:13px; color:#e6edf3; margin-top:5px;">
                {body}
            </div>
            {"<div style='font-size:11px; color:#8b949e; margin-top:6px;'>" + detail + "</div>" if detail else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Individual alert checks ───────────────────────────────────

def check_power_threshold(df: pd.DataFrame) -> list[dict]:
    """Return list of alert dicts for any server breaching power thresholds."""
    alerts = []
    if df.empty or "Power_Usage_Watts" not in df.columns:
        return alerts

    for server_id, sdf in df.groupby("Server_ID"):
        max_power = float(sdf["Power_Usage_Watts"].max())
        latest_power = float(sdf.sort_values("Timestamp").iloc[-1]["Power_Usage_Watts"]) \
            if "Timestamp" in sdf.columns else max_power
        latest_ts = str(sdf.sort_values("Timestamp").iloc[-1]["Timestamp"]) \
            if "Timestamp" in sdf.columns else "—"

        if max_power >= ALERT_POWER_CRITICAL_W:
            alerts.append({
                "level": "critical",
                "title": f"⚡ Critical Power Spike — {server_id}",
                "body": (
                    f"Server <b>{server_id}</b> reached <b>{max_power:.1f}W</b>, "
                    f"exceeding the critical threshold of <b>{ALERT_POWER_CRITICAL_W}W</b>."
                ),
                "detail": f"Latest reading: {latest_power:.1f}W at {latest_ts}",
            })
        elif max_power >= ALERT_POWER_WARNING_W:
            alerts.append({
                "level": "warning",
                "title": f"⚡ Power Warning — {server_id}",
                "body": (
                    f"Server <b>{server_id}</b> reached <b>{max_power:.1f}W</b>, "
                    f"exceeding the warning threshold of <b>{ALERT_POWER_WARNING_W}W</b>."
                ),
                "detail": f"Latest reading: {latest_power:.1f}W at {latest_ts}",
            })

    return alerts


def check_carbon_threshold(df: pd.DataFrame) -> list[dict]:
    """Return alert if total fleet carbon exceeds threshold."""
    alerts = []
    if df.empty or "carbon_kg" not in df.columns:
        return alerts

    total_carbon = float(df["carbon_kg"].sum())
    if total_carbon >= ALERT_CARBON_CRITICAL_KG:
        alerts.append({
            "level": "critical",
            "title": "🌍 Carbon Emission Threshold Breached",
            "body": (
                f"Total fleet carbon emissions reached <b>{total_carbon:.3f} kg CO₂</b>, "
                f"exceeding the critical threshold of <b>{ALERT_CARBON_CRITICAL_KG} kg</b>."
            ),
            "detail": f"Highest emitting server: "
                      f"{df.groupby('Server_ID')['carbon_kg'].sum().idxmax()}",
        })
    return alerts


def check_anomaly_threshold(anomaly_df: pd.DataFrame) -> list[dict]:
    """Return alert if anomaly count exceeds thresholds."""
    alerts = []
    total = len(anomaly_df)
    critical_count = int((anomaly_df.get("severity_label", pd.Series()) == "Critical").sum()) \
        if not anomaly_df.empty else 0

    if total >= ALERT_ANOMALY_CRITICAL:
        alerts.append({
            "level": "critical",
            "title": "🔍 High Anomaly Count Detected",
            "body": (
                f"<b>{total}</b> anomalies detected across the fleet "
                f"(threshold: <b>{ALERT_ANOMALY_CRITICAL}</b>). "
                f"<b>{critical_count}</b> are Critical severity."
            ),
            "detail": "Go to Anomaly Detection page for full details.",
        })
    elif total >= ALERT_ANOMALY_WARNING:
        alerts.append({
            "level": "warning",
            "title": "🔍 Elevated Anomaly Count",
            "body": (
                f"<b>{total}</b> anomalies detected "
                f"(warning threshold: <b>{ALERT_ANOMALY_WARNING}</b>)."
            ),
            "detail": "Monitor closely. Go to Anomaly Detection page for details.",
        })
    return alerts


def check_per_server_carbon(df: pd.DataFrame) -> list[dict]:
    """Warn if any individual server's carbon is disproportionately high."""
    alerts = []
    if df.empty or "carbon_kg" not in df.columns:
        return alerts

    per_server = df.groupby("Server_ID")["carbon_kg"].sum()
    fleet_mean = float(per_server.mean())
    fleet_std  = float(per_server.std(ddof=0)) if len(per_server) > 1 else 0.0

    for server_id, carbon in per_server.items():
        if fleet_std > 0 and (carbon - fleet_mean) / fleet_std > 2.0:
            alerts.append({
                "level": "warning",
                "title": f"🌍 Carbon Outlier — {server_id}",
                "body": (
                    f"Server <b>{server_id}</b> emitted <b>{carbon:.3f} kg CO₂</b>, "
                    f"more than 2 standard deviations above the fleet average "
                    f"(<b>{fleet_mean:.3f} kg</b>)."
                ),
                "detail": "Consider workload rebalancing or shifting to a greener window.",
            })
    return alerts


# ── Main entry point ──────────────────────────────────────────

def evaluate_and_show_alerts(location: str = "inline") -> int:
    """
    Run all alert checks and render results.

    Args:
        location: "inline" — renders alerts directly where called (use in sidebar or page top)

    Returns:
        int — total number of alerts fired (0 = all clear)
    """
    raw_df     = _raw_df()
    anomaly_df = _anomaly_df()

    all_alerts = []
    all_alerts.extend(check_power_threshold(raw_df))
    all_alerts.extend(check_carbon_threshold(raw_df))
    all_alerts.extend(check_anomaly_threshold(anomaly_df))
    all_alerts.extend(check_per_server_carbon(raw_df))

    if not all_alerts:
        return 0

    # Sort: critical first, then warning, then info
    order = {"critical": 0, "warning": 1, "info": 2}
    all_alerts.sort(key=lambda a: order.get(a["level"], 2))

    # Count by level
    n_critical = sum(1 for a in all_alerts if a["level"] == "critical")
    n_warning  = sum(1 for a in all_alerts if a["level"] == "warning")

    # ── Header summary popup bar ──────────────────────────────
    summary_parts = []
    if n_critical:
        summary_parts.append(f"🔴 {n_critical} Critical")
    if n_warning:
        summary_parts.append(f"🟡 {n_warning} Warning")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #3d1a1a, #1a1a2e);
            border: 1px solid #f85149;
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size:20px;">🚨</span>
            <span style="font-size:14px; font-weight:700; color:#f85149;">
                ALERT: {" &nbsp;|&nbsp; ".join(summary_parts)} — {len(all_alerts)} active alert{"s" if len(all_alerts) != 1 else ""}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Individual alert banners inside expander ──────────────
    with st.expander(f"View {len(all_alerts)} Alert Detail{'s' if len(all_alerts) != 1 else ''}", expanded=n_critical > 0):
        for alert in all_alerts:
            _render_alert(
                level=alert["level"],
                title=alert["title"],
                body=alert["body"],
                detail=alert.get("detail", ""),
            )

    return len(all_alerts)


def show_sidebar_alert_badge() -> None:
    """
    Show a compact alert count badge in the sidebar.
    Call this inside the `with st.sidebar:` block.
    """
    raw_df     = _raw_df()
    anomaly_df = _anomaly_df()

    all_alerts = (
        check_power_threshold(raw_df)
        + check_carbon_threshold(raw_df)
        + check_anomaly_threshold(anomaly_df)
        + check_per_server_carbon(raw_df)
    )

    n_critical = sum(1 for a in all_alerts if a["level"] == "critical")
    n_warning  = sum(1 for a in all_alerts if a["level"] == "warning")

    if n_critical:
        st.markdown(
            f"<div style='background:#3d1a1a;border:1px solid #f85149;border-radius:8px;"
            f"padding:8px 12px;font-size:13px;color:#f85149;font-weight:700;'>"
            f"🔴 {n_critical} Critical Alert{'s' if n_critical != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
    if n_warning:
        st.markdown(
            f"<div style='background:#2e1f00;border:1px solid #d29922;border-radius:8px;"
            f"padding:8px 12px;font-size:13px;color:#d29922;font-weight:700;margin-top:6px;'>"
            f"🟡 {n_warning} Warning{'s' if n_warning != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
    if not all_alerts:
        st.markdown(
            "<div style='background:#0d2e1a;border:1px solid #39d353;border-radius:8px;"
            "padding:8px 12px;font-size:13px;color:#39d353;font-weight:600;'>"
            "✅ All systems normal</div>",
            unsafe_allow_html=True,
        )