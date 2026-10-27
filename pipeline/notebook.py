# Databricks notebook source
# MAGIC %md
# MAGIC # AI value tracking pipeline (bronze → silver → gold)
# MAGIC Monthly actuals files from the plant land in S3, are synced into the `landing` volume, and flow through:
# MAGIC - **bronze_actuals**: every row exactly as received (all text), with the source file and load time. `COPY INTO` skips files already loaded.
# MAGIC - **silver_actuals**: typed rows that pass the same checks as the app's Track value tab; a corrected file for a month replaces the earlier one.
# MAGIC - **quarantine_actuals**: rows that failed, with the reason, so the plant can fix and resend them.
# MAGIC - **gold_tracked_value**: monthly forecast vs actual savings, adoption and drift flags (same formula as `src/tracking.py`).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ai_roi")
for name, default in {"cost_missed": "25000", "cost_caught": "4000", "cost_false_alarm": "500",
                      "running_cost": "60000", "annual_net": "0", "launch_recall": "0"}.items():
    dbutils.widgets.text(name, default)

CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
P = {k: float(dbutils.widgets.get(k)) for k in
     ["cost_missed", "cost_caught", "cost_false_alarm", "running_cost", "annual_net", "launch_recall"]}
ADOPTION_FLOOR, DRIFT_DROP = 0.70, 0.10          # same thresholds as src/tracking.py
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## Bronze: load new files as received

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS bronze_actuals (
  month STRING, alerts_raised STRING, failures_flagged STRING, failures_prevented STRING,
  failures_missed STRING, false_alarms STRING, source_file STRING, ingested_at TIMESTAMP)
""")
loaded = spark.sql(f"""
COPY INTO bronze_actuals
FROM (SELECT month, alerts_raised, failures_flagged, failures_prevented, failures_missed, false_alarms,
             _metadata.file_path AS source_file, current_timestamp() AS ingested_at
      FROM '{VOLUME}/actuals/')
FILEFORMAT = CSV
PATTERN = '*/*.csv'
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
""").collect()
display(loaded)

# COMMAND ----------

# MAGIC %md ## Silver and quarantine: type, check, keep the latest file per month

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW checked AS
WITH typed AS (
  SELECT month, source_file, ingested_at,
         try_cast(alerts_raised AS INT) AS alerts_raised, try_cast(failures_flagged AS INT) AS failures_flagged,
         try_cast(failures_prevented AS INT) AS failures_prevented, try_cast(failures_missed AS INT) AS failures_missed,
         try_cast(false_alarms AS INT) AS false_alarms
  FROM bronze_actuals
)
SELECT *, filter(array(
    CASE WHEN month IS NULL OR NOT month RLIKE '^[0-9]{4}-(0[1-9]|1[0-2])$' THEN 'month must look like 2026-01' END,
    CASE WHEN alerts_raised IS NULL OR failures_flagged IS NULL OR failures_prevented IS NULL
              OR failures_missed IS NULL OR false_alarms IS NULL THEN 'every count must be a number' END,
    CASE WHEN least(alerts_raised, failures_flagged, failures_prevented, failures_missed, false_alarms) < 0
         THEN 'counts cannot be negative' END,
    CASE WHEN failures_prevented > failures_flagged THEN 'failures_prevented cannot exceed failures_flagged' END,
    CASE WHEN failures_flagged > failures_prevented + failures_missed
         THEN 'failures_flagged cannot exceed total failures' END,
    CASE WHEN failures_prevented + false_alarms > alerts_raised
         THEN 'alerts acted on cannot exceed alerts_raised' END
  ), r -> r IS NOT NULL) AS reasons
FROM typed
""")
spark.sql("""
CREATE OR REPLACE TABLE quarantine_actuals AS
SELECT month, source_file, ingested_at, array_join(reasons, '; ') AS reason FROM checked WHERE size(reasons) > 0
""")
spark.sql("""
CREATE OR REPLACE TABLE silver_actuals AS
SELECT month, alerts_raised, failures_flagged, failures_prevented, failures_missed, false_alarms, source_file, ingested_at
FROM checked WHERE size(reasons) = 0
QUALIFY row_number() OVER (PARTITION BY month ORDER BY ingested_at DESC, source_file DESC) = 1
""")

# COMMAND ----------

# MAGIC %md ## Gold: forecast vs actual savings, adoption and drift

# COMMAND ----------

G = {**P, "adoption_floor": ADOPTION_FLOOR, "drift_drop": DRIFT_DROP}   # all floats, so safe to format into SQL
spark.sql(f"""
CREATE OR REPLACE TABLE gold_tracked_value AS
WITH base AS (
  SELECT *, failures_prevented + failures_missed AS failures, failures_prevented + false_alarms AS acted_on,
         {G['annual_net']} / 12 AS forecast,
         failures_prevented * ({G['cost_missed']} - {G['cost_caught']}) - false_alarms * {G['cost_false_alarm']} - {G['running_cost']} / 12 AS actual
  FROM silver_actuals
), rolled AS (
  SELECT *,
         failures_flagged / nullif(failures, 0) AS recall,
         acted_on / nullif(alerts_raised, 0) AS adoption,
         CASE WHEN count(*) OVER w3 = 3
              THEN sum(failures_flagged) OVER w3 / nullif(sum(failures) OVER w3, 0) END AS recall_3m,
         sum(forecast) OVER (ORDER BY month) AS cum_forecast,
         sum(actual) OVER (ORDER BY month) AS cum_actual
  FROM base
  WINDOW w3 AS (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
)
SELECT month, alerts_raised, failures_flagged, failures_prevented, failures_missed, false_alarms,
       failures, acted_on, recall, adoption, recall_3m, forecast, actual, cum_forecast, cum_actual,
       coalesce(adoption < {G['adoption_floor']}, false) AS adoption_alert,
       coalesce(recall_3m < {G['launch_recall']} - {G['drift_drop']}, false) AS drift_alert,
       source_file
FROM rolled ORDER BY month
""")
display(spark.table("gold_tracked_value"))

# COMMAND ----------

# MAGIC %md ## Export for the app (no Databricks access needed to view it)

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

os.makedirs(f"{VOLUME}/exports", exist_ok=True)
spark.table("gold_tracked_value").toPandas().to_csv(f"{VOLUME}/exports/tracked_value.csv", index=False)
q = spark.table("quarantine_actuals").toPandas()
run_log = {
    "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "rows_loaded_this_run": int(loaded[0]["num_inserted_rows"]) if loaded else 0,
    "bronze_rows": spark.table("bronze_actuals").count(),
    "silver_rows": spark.table("silver_actuals").count(),
    "quarantined": [{"month": r.month, "file": r.source_file.split("/actuals/")[-1], "reason": r.reason} for r in q.itertuples()],
    "parameters": P,
}
with open(f"{VOLUME}/exports/run_log.json", "w") as f:
    json.dump(run_log, f, indent=1)
print(json.dumps(run_log, indent=1))
