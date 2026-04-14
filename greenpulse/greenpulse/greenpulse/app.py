import time
from datetime import datetime

import pandas as pd
import streamlit as st

from alerts import evaluate_and_show_alerts, show_sidebar_alert_badge
from config import CSV_PATH
from database.mongo import (
    get_all_anomalies,
    get_all_raw_data,
    get_servers_list,
    get_shift_results,
    ping_db,
)
from features.anomaly import run_anomaly_detection
from features.prediction import run_carbon_prediction
from features.shifting import run_shift_simulation
from pages.anomaly_detection import render_anomaly_page
from pages.dashboard import render_dashboard
from pages.carbon_prediction import render_prediction_page
from pages.live_monitor import render_live_monitor
from pages.server_details import render_server_details
from pages.workload_shifting import render_shifting_page
from pipeline.ingest import ingest_csv_to_mongo

st.set_page_config(
    page_title="GreenPulse - Carbon Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.raw_data = []
    st.session_state.anomaly_data = []
    st.session_state.shift_results = {}
    st.session_state.prediction_results = {}
    st.session_state.last_refresh = None
    st.session_state.db_connected = False
    st.session_state.selected_server = None
    st.session_state.alerts_dismissed = False
    st.session_state.alert_threshold_power = 160
    st.session_state.live_paused = False

st.markdown(
    """
<style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
    }
    .stMarkdown, .stText { color: #e6edf3; }
    .metric-positive { color: #39d353; }
    .metric-alert    { color: #f85149; }
    .metric-warning  { color: #d29922; }
    .custom-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .dataframe { background-color: #161b22; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_cached_data() -> tuple[list[dict], list[dict], dict, list[str]]:
    raw = get_all_raw_data()
    anomalies = get_all_anomalies()
    shift = get_shift_results() or {}
    servers = get_servers_list()
    return raw, anomalies, shift, servers


@st.cache_resource
def cached_ping() -> bool:
    return ping_db()


def sync_session_from_db() -> None:
    raw, anomalies, shift, servers = load_cached_data()
    st.session_state.raw_data = raw
    st.session_state.anomaly_data = anomalies
    st.session_state.shift_results = shift
    if servers and not st.session_state.selected_server:
        st.session_state.selected_server = servers[0]


def run_full_pipeline() -> None:
    with st.spinner("Initializing GreenPulse..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("📁 Loading and cleaning data...")
            progress_bar.progress(20)
            ingest_csv_to_mongo(CSV_PATH)

            status_text.text("🔍 Running anomaly detection...")
            progress_bar.progress(50)
            raw_df = pd.DataFrame(get_all_raw_data())
            if not raw_df.empty:
                raw_df["Timestamp"] = pd.to_datetime(raw_df["Timestamp"], errors="coerce")
                processed_df = run_anomaly_detection(raw_df)
            else:
                processed_df = raw_df

            status_text.text("⚡ Simulating workload shifting...")
            progress_bar.progress(65)
            run_shift_simulation(processed_df)

            status_text.text("🤖 Training carbon prediction model...")
            progress_bar.progress(85)
            pred_results = run_carbon_prediction(processed_df)
            st.session_state.prediction_results = pred_results

            status_text.text("✅ Complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            load_cached_data.clear()
            sync_session_from_db()
            st.session_state.last_refresh = datetime.now()
            # Reset alert dismissal on fresh data load
            st.session_state.alerts_dismissed = False
            st.success("GreenPulse data pipeline completed successfully.")
        except Exception as exc:
            st.error(f"Pipeline execution failed: {exc}")


st.session_state.db_connected = cached_ping()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## <span style='color:#39d353;'>GreenPulse</span>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard (Fleet Overview)",
            "🔴 Live Monitor",
            "🔍 Anomaly Detection",
            "⚡ Workload Shifting",
            "🤖 Carbon Prediction",
            "📊 Server Details",
        ],
    )

    st.markdown("---")
    st.subheader("Data Management")
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh_clicked:
        cached_ping.clear()
        st.session_state.db_connected = cached_ping()
    db_icon = "🟢" if st.session_state.db_connected else "🔴"
    st.write(f"{db_icon} Database Connected: {st.session_state.db_connected}")
    st.write(
        "Last Updated:",
        st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S")
        if st.session_state.last_refresh
        else "Never",
    )

    st.markdown("---")
    st.subheader("Quick Stats")
    total_servers = len({r.get("Server_ID") for r in st.session_state.raw_data if r.get("Server_ID")})
    total_records = len(st.session_state.raw_data)
    total_anomalies = len(st.session_state.anomaly_data)
    st.write(f"Total Servers: {total_servers}")
    st.write(f"Total Records: {total_records}")
    st.write(f"Total Anomalies: {total_anomalies}")

    # ── Alert threshold control in sidebar ───────────────────
    st.markdown("---")
    st.subheader("🚨 Alert Settings")
    st.session_state.alert_threshold_power = st.slider(
        "Power Warning Threshold (W)",
        min_value=100,
        max_value=200,
        value=int(st.session_state.alert_threshold_power),
        step=5,
        help="Show warning alert when any server exceeds this wattage",
    )
    if st.button("🔕 Dismiss Alerts", use_container_width=True):
        st.session_state.alerts_dismissed = True
        st.rerun()
    if st.session_state.alerts_dismissed:
        if st.button("🔔 Re-enable Alerts", use_container_width=True):
            st.session_state.alerts_dismissed = False
            st.rerun()

    # ── Alert badge in sidebar ────────────────────────────────
    st.markdown("---")
    if st.session_state.raw_data and not st.session_state.alerts_dismissed:
        show_sidebar_alert_badge()

# ── DB check ─────────────────────────────────────────────────
if not st.session_state.db_connected:
    st.warning("MongoDB is not connected. Please start MongoDB at mongodb://localhost:27017/ and refresh.")
else:
    should_initialize = not st.session_state.initialized
    try:
        if should_initialize:
            existing_raw = get_all_raw_data()
            if len(existing_raw) == 0:
                run_full_pipeline()
            else:
                sync_session_from_db()
                st.session_state.last_refresh = datetime.now()
            st.session_state.initialized = True

        if refresh_clicked:
            run_full_pipeline()
            st.session_state.initialized = True
    except Exception as exc:
        st.error(f"Initialization error: {exc}")

# ── Alert banner (top of every page, before page content) ────
if st.session_state.raw_data and not st.session_state.alerts_dismissed:
    evaluate_and_show_alerts()

# ── Page routing ──────────────────────────────────────────────
if page == "🏠 Dashboard (Fleet Overview)":
    render_dashboard()
elif page == "🔴 Live Monitor":
    render_live_monitor()
elif page == "🔍 Anomaly Detection":
    render_anomaly_page()
elif page == "⚡ Workload Shifting":
    render_shifting_page()
elif page == "🤖 Carbon Prediction":
    render_prediction_page()
elif page == "📊 Server Details":
    render_server_details()