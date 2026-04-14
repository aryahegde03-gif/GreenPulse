from typing import Any

from pymongo import MongoClient

from config import (
    COLLECTION_ANOMALIES,
    COLLECTION_RAW,
    COLLECTION_SHIFTS,
    DB_NAME,
    MONGO_URI,
)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
raw_col = db[COLLECTION_RAW]
anomaly_col = db[COLLECTION_ANOMALIES]
shift_col = db[COLLECTION_SHIFTS]


def _serialize_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for doc in docs:
        clean_doc = {k: v for k, v in doc.items() if k != "_id"}
        serialized.append(clean_doc)
    return serialized


def ping_db() -> bool:
    try:
        client.admin.command("ping")
        return True
    except Exception as exc:
        print(f"[MONGO] Ping failed: {exc}")
        return False


def get_all_raw_data() -> list[dict[str, Any]]:
    try:
        docs = list(raw_col.find({}))
        return _serialize_docs(docs)
    except Exception as exc:
        print(f"[MONGO] Failed to read raw data: {exc}")
        return []


def get_all_anomalies() -> list[dict[str, Any]]:
    try:
        docs = list(anomaly_col.find({}))
        return _serialize_docs(docs)
    except Exception as exc:
        print(f"[MONGO] Failed to read anomalies: {exc}")
        return []


def get_shift_results() -> dict[str, Any] | None:
    try:
        doc = shift_col.find_one({})
        if not doc:
            return None
        doc.pop("_id", None)
        return doc
    except Exception as exc:
        print(f"[MONGO] Failed to read shift results: {exc}")
        return None


def get_servers_list() -> list[str]:
    try:
        servers = raw_col.distinct("Server_ID")
        return sorted([s for s in servers if s is not None])
    except Exception as exc:
        print(f"[MONGO] Failed to fetch server list: {exc}")
        return []


def get_server_data(server_id: str) -> list[dict[str, Any]]:
    try:
        docs = list(raw_col.find({"Server_ID": server_id}))
        return _serialize_docs(docs)
    except Exception as exc:
        print(f"[MONGO] Failed to read server data for {server_id}: {exc}")
        return []


def get_server_anomalies(server_id: str) -> list[dict[str, Any]]:
    try:
        docs = list(anomaly_col.find({"Server_ID": server_id}))
        return _serialize_docs(docs)
    except Exception as exc:
        print(f"[MONGO] Failed to read anomalies for {server_id}: {exc}")
        return []
