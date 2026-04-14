from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "greenpulse"
COLLECTION_RAW = "server_emissions"
COLLECTION_ANOMALIES = "anomalies"
COLLECTION_SHIFTS = "shift_recommendations"
EMISSION_FACTOR = 0.82
CSV_PATH = str(_BASE_DIR / "dataset.csv")

# ── Alert thresholds ──────────────────────────────────────────
ALERT_POWER_CRITICAL_W   = 180   # watts — single reading critical spike
ALERT_POWER_WARNING_W    = 150   # watts — single reading warning
ALERT_CARBON_CRITICAL_KG = 500   # kg CO₂ — total fleet carbon critical
ALERT_ANOMALY_CRITICAL   = 100   # anomaly count — critical threshold
ALERT_ANOMALY_WARNING    = 50    # anomaly count — warning threshold

PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor": "#161b22",
        "font": {"color": "#e6edf3"},
        "xaxis": {"gridcolor": "#30363d", "linecolor": "#30363d"},
        "yaxis": {"gridcolor": "#30363d", "linecolor": "#30363d"},
    }
}