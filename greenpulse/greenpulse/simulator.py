"""
Real-time server emission simulator.
Streams dataset.csv row by row into MongoDB every few seconds,
mimicking live server telemetry data coming in.

Run this in a SEPARATE terminal:
    python simulator.py
"""

import sys
import time
import random
from datetime import datetime

sys.path.insert(0, "greenpulse")

import pandas as pd
from database.mongo import raw_col, anomaly_col
from pipeline.clean import clean_dataframe
from config import EMISSION_FACTOR

INTERVAL_SECONDS = 3   # New reading every 3 seconds
SERVERS = ["S1", "S2", "S3"]
BASE_POWER = {"S1": 110, "S2": 135, "S3": 115}  # Typical watts per server


def generate_live_reading(server_id: str, add_spike: bool = False) -> dict:
    """Generate a synthetic realistic power reading for a server."""
    base = BASE_POWER[server_id]

    if add_spike:
        # Inject anomaly spike
        power = base + random.uniform(80, 150)
    else:
        # Normal variation ±20%
        noise = random.gauss(0, base * 0.12)
        power = max(50, base + noise)

    power = round(power, 2)
    now = datetime.now()
    energy_kwh = round((power * (INTERVAL_SECONDS / 3600)) / 1000, 8)
    carbon_kg = round(energy_kwh * EMISSION_FACTOR, 8)

    return {
        "Timestamp": now,
        "Server_ID": server_id,
        "Power_Usage_Watts": power,
        "energy_kwh": energy_kwh,
        "carbon_kg": carbon_kg,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "hours_interval": INTERVAL_SECONDS / 3600,
        "is_live": True,
    }


def run_simulator():
    print("=" * 50)
    print("  GreenPulse Real-Time Emission Simulator")
    print(f"  Generating readings every {INTERVAL_SECONDS}s")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    reading_count = 0
    spike_every = 20  # Inject an anomaly spike every 20 readings

    while True:
        reading_count += 1
        batch = []

        for server_id in SERVERS:
            # Occasionally spike one server to simulate anomaly
            add_spike = (reading_count % spike_every == 0) and (server_id == "S2")
            record = generate_live_reading(server_id, add_spike=add_spike)
            batch.append(record)

            tag = "💥 SPIKE!" if add_spike else "✅"
            print(
                f"  {tag} {record['Timestamp'].strftime('%H:%M:%S')} | "
                f"{server_id} | {record['Power_Usage_Watts']:6.1f}W | "
                f"{record['carbon_kg']:.6f} kg CO₂"
            )

        # Insert into MongoDB
        raw_col.insert_many(batch)

        total_carbon = sum(r["carbon_kg"] for r in batch)
        total_power = sum(r["Power_Usage_Watts"] for r in batch)
        print(f"  📊 Fleet total: {total_power:.1f}W  |  {total_carbon:.6f} kg CO₂")
        print(f"  📝 Total readings in DB: {raw_col.count_documents({'is_live': True})}")
        print()

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run_simulator()
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
