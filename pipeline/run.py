"""Value-tracking pipeline: S3 landing zone -> Databricks (bronze/silver/gold) -> app.

    land    split the sample actuals into one file per month (plus one bad file) and upload them to S3,
            as the plant's maintenance system would
    sync    copy new S3 files into the Databricks `landing` volume
    run     upload pipeline/notebook.py and run it as a Databricks job (serverless)
    fetch   download the gold export to data/pipeline/ and archive it in S3 under curated/

Databricks Free Edition cannot read your own S3 bucket (no external locations), so `sync` is the bridge.
With paid Databricks, replace it with a Unity Catalog external location on the bucket and let COPY INTO
read s3:// directly.

Usage (AWS sign-in via `aws login`, Databricks sign-in opens a browser the first time):
    uv run --no-project --python 3.12 --with-requirements requirements.txt --with "boto3[crt]" --with databricks-sdk \
        python -m pipeline.run land sync run fetch --aws-profile AI-ROI
"""

import argparse
import base64
import io
import json
import os

import pandas as pd

from src.model import train_and_score
from src.value import Costs, evaluate, sweep

BUCKET = "ai-roi-data-138625638112-us-east-2-an"
REGION = "us-east-2"
DATABRICKS_HOST = "https://dbc-788105af-5195.cloud.databricks.com"
CATALOG, SCHEMA, VOLUME = "workspace", "ai_roi", "landing"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
JOB_NAME = "ai-roi-value-pipeline"
OUT = "data/pipeline"


def s3_client(profile):
    import boto3
    return boto3.Session(profile_name=profile, region_name=REGION).client("s3")


def workspace():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient(host=DATABRICKS_HOST, auth_type="external-browser")


def land(s3):
    """One file per month, as a plant would send them, plus a file with a data-entry error."""
    df = pd.read_csv("data/sample_actuals.csv")
    for _, row in df.iterrows():
        key = f"landing/actuals/month={row['month']}/actuals.csv"
        s3.put_object(Bucket=BUCKET, Key=key, Body=row.to_frame().T.to_csv(index=False).encode())
    bad = pd.DataFrame([{"month": "2027-01", "alerts_raised": 3, "failures_flagged": 2, "failures_prevented": 4,
                         "failures_missed": 1, "false_alarms": 0}])          # prevented > flagged: impossible
    s3.put_object(Bucket=BUCKET, Key="landing/actuals/month=2027-01/actuals.csv", Body=bad.to_csv(index=False).encode())
    print(f"Landed {len(df) + 1} files in s3://{BUCKET}/landing/actuals/")


def sync(s3, w):
    """Copy S3 landing files that are not yet in the Databricks volume."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.catalog import VolumeType
    if SCHEMA not in [s.name for s in w.schemas.list(catalog_name=CATALOG)]:
        w.schemas.create(name=SCHEMA, catalog_name=CATALOG, comment="AI ROI Navigator value tracking")
    if VOLUME not in [v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)]:
        w.volumes.create(catalog_name=CATALOG, schema_name=SCHEMA, name=VOLUME, volume_type=VolumeType.MANAGED)
    copied = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="landing/"):
        for obj in page.get("Contents", []):
            target = f"{VOLUME_PATH}/{obj['Key'].removeprefix('landing/')}"
            try:
                w.files.get_metadata(target)
                continue
            except NotFound:
                pass
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
            w.files.upload(target, io.BytesIO(body), overwrite=False)
            copied += 1
    print(f"Synced {copied} new files to {VOLUME_PATH}")


def forecast() -> dict:
    """The forecast the business case was approved on: gradient boosting at the dollar-optimal threshold."""
    c = Costs()
    res = train_and_score()
    y, p = res["y_test"], res["models"]["Gradient boosting"]["proba"]
    curve = sweep(y, p, c)
    r = evaluate(y, p, float(curve.loc[curve["annual_net"].idxmax(), "threshold"]), c)
    return {"cost_missed": c.cost_missed, "cost_caught": c.cost_caught, "cost_false_alarm": c.cost_false_alarm,
            "running_cost": c.running_cost, "annual_net": round(r["annual_net"], 2), "launch_recall": round(r["recall"], 6)}


def run(w):
    from databricks.sdk.service import jobs, workspace as ws
    me = w.current_user.me().user_name
    path = f"/Users/{me}/ai_roi_pipeline"
    with open("pipeline/notebook.py", "rb") as f:
        w.workspace.import_(path=path, format=ws.ImportFormat.SOURCE, language=ws.Language.PYTHON,
                            content=base64.b64encode(f.read()).decode(), overwrite=True)
    params = {k: str(v) for k, v in forecast().items()}
    task = jobs.Task(task_key="bronze_silver_gold",
                     notebook_task=jobs.NotebookTask(notebook_path=path, base_parameters=params))
    schedule = jobs.CronSchedule(quartz_cron_expression="0 0 6 2 * ?", timezone_id="UTC",   # 2nd of each month
                                 pause_status=jobs.PauseStatus.PAUSED)
    existing = [j for j in w.jobs.list(name=JOB_NAME)]
    if existing:
        job_id = existing[0].job_id
        w.jobs.reset(job_id, jobs.JobSettings(name=JOB_NAME, tasks=[task], schedule=schedule, max_concurrent_runs=1))
    else:
        job_id = w.jobs.create(name=JOB_NAME, tasks=[task], schedule=schedule, max_concurrent_runs=1).job_id
    print(f"Running job {job_id} ...")
    result = w.jobs.run_now(job_id).result()
    print("Result:", result.state.result_state, result.run_page_url)


def fetch(s3, w):
    os.makedirs(OUT, exist_ok=True)
    for name in ("tracked_value.csv", "run_log.json"):
        data = w.files.download(f"{VOLUME_PATH}/exports/{name}").contents.read()
        with open(f"{OUT}/{name}", "wb") as f:
            f.write(data)
        s3.put_object(Bucket=BUCKET, Key=f"curated/tracked_value/{name}", Body=data)
    log = json.load(open(f"{OUT}/run_log.json"))
    print(f"Fetched gold export ({log['silver_rows']} months, {len(log['quarantined'])} quarantined) to {OUT}/ "
          f"and archived it in s3://{BUCKET}/curated/tracked_value/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["land", "sync", "run", "fetch"])
    ap.add_argument("--aws-profile", default=None)
    args = ap.parse_args()
    s3 = s3_client(args.aws_profile)
    w = workspace() if set(args.steps) - {"land"} else None
    for step in args.steps:
        {"land": lambda: land(s3), "sync": lambda: sync(s3, w), "run": lambda: run(w), "fetch": lambda: fetch(s3, w)}[step]()
