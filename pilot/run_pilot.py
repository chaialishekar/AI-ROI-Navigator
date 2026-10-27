"""Run the work-order pilot on Amazon Bedrock and save raw results.

Usage (needs AWS credentials with bedrock:InvokeModel on Nova Micro in us-east-2):
    uv run --no-project --python 3.12 --with-requirements requirements.txt --with boto3 \
        python -m pilot.run_pilot --profile AI-ROI --version 2

Writes pilot/pilot_results_v{1,2}.csv: one row per case with the reply, token counts and latency.
Version 1: the model diagnoses and writes. Version 2: code diagnoses (src.workorder.diagnose), the model writes.
Cost: 30 calls at ~$0.0001 each.
"""

import argparse
import time

import boto3
import pandas as pd
from botocore.config import Config

from src.workorder import MAX_TOKENS, MODEL_ID, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, diagnose, user_prompt

REGION = "us-east-2"


def run(cases: pd.DataFrame, client, version: int = 1) -> pd.DataFrame:
    system = SYSTEM_PROMPT if version == 1 else SYSTEM_PROMPT_V2
    rows = []
    for _, case in cases.iterrows():
        start = time.perf_counter()
        resp = client.converse(
            modelId=MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_prompt(case, version)}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0},
        )
        rows.append({
            "case_id": case["case_id"],
            "true_mode": case["true_mode"],
            **({"given_mode": diagnose(case)[0]} if version == 2 else {}),
            "reply": resp["output"]["message"]["content"][0]["text"],
            "stop_reason": resp["stopReason"],
            "input_tokens": resp["usage"]["inputTokens"],
            "output_tokens": resp["usage"]["outputTokens"],
            "latency_ms": resp["metrics"]["latencyMs"],
            "wall_ms": round((time.perf_counter() - start) * 1000),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--version", type=int, default=2, choices=[1, 2])
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    # Retry only throttling-type errors; access and validation errors should stop the run.
    client = session.client("bedrock-runtime", config=Config(retries={"max_attempts": 5, "mode": "adaptive"}))
    out = run(pd.read_csv("pilot/pilot_cases.csv"), client, args.version)
    path = f"pilot/pilot_results_v{args.version}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} results to {path}")
