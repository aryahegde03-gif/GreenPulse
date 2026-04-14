import pandas as pd

from database.mongo import raw_col
from pipeline.clean import clean_dataframe


def ingest_csv_to_mongo(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
        cleaned_df = clean_dataframe(df)

        raw_col.delete_many({})
        records = cleaned_df.to_dict("records")

        for record in records:
            if isinstance(record.get("Timestamp"), pd.Timestamp):
                record["Timestamp"] = record["Timestamp"].to_pydatetime()

        if records:
            raw_col.insert_many(records)

        summary = {
            "rows_inserted": len(records),
            "servers_found": sorted(cleaned_df["Server_ID"].unique().tolist()),
            "time_range_start": str(cleaned_df["Timestamp"].min()),
            "time_range_end": str(cleaned_df["Timestamp"].max()),
        }
        print(f"[INGEST] {summary['rows_inserted']} rows inserted into MongoDB")
        return summary
    except Exception as exc:
        print(f"[INGEST] Failed to ingest CSV: {exc}")
        raise
