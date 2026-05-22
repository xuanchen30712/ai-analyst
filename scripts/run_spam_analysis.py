#!/usr/bin/env python
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from helpers.data_helpers import read_table
from helpers.analytics_helpers import (
compare_segments,
concentration_analysis,
score_findings,
synthesize_insights,
)

def norm_num(x):
    s = re.sub(r"\D", "", str(x))
    return s[-10:] if len(s) >= 10 else s

def main():
    # 1) Load tables
    t1 = read_table("table1").copy()
    t2 = read_table("table2").copy()

    # 2) Normalize join keys
    t1["num_norm"] = t1["calling_number"].map(norm_num)
    t2["num_norm"] = t2["Calling Party No."].map(norm_num)

    t1 = t1[t1["num_norm"] != ""].drop_duplicates("num_norm")
    t2 = t2[t2["num_norm"] != ""]

    # 3) Flag risk numbers in CDR
    risk_set = set(t1["num_norm"])
    t2["is_risk"] = t2["num_norm"].isin(risk_set)

    # 4) Parse metrics
    t2["Duration_num"] = pd.to_numeric(t2["Duration"], errors="coerce").fillna(0)

    pat16 = re.compile(r"(^|\D)16(\D|$)")
    has16 = lambda x: bool(pat16.search(str(x)))
    t2["cause16"] = t2["Originator Causes"].map(has16) | t2["Destination Causes"].map(has16)

    # 5) Core coverage + impact metrics
    unique_t1 = t1["num_norm"].nunique()
    unique_t2 = t2["num_norm"].nunique()
    overlap = t2.loc[t2["is_risk"], "num_norm"].nunique()
    overlap_pct_of_t1 = overlap / unique_t1 if unique_t1 else 0
    overlap_pct_of_t2 = overlap / unique_t2 if unique_t2 else 0

    total_calls = len(t2)
    risk_calls = int(t2["is_risk"].sum())
    risk_call_share = risk_calls / total_calls if total_calls else 0

    risk_cause16_share = t2.loc[t2["is_risk"], "cause16"].mean() if risk_calls else 0
    nonrisk_cause16_share = t2.loc[~t2["is_risk"], "cause16"].mean() if (total_calls - risk_calls) else 0

    print("\n=== LINKAGE + COVERAGE ===")
    print("Risk numbers in table1:", unique_t1)
    print("Unique calling numbers in table2:", unique_t2)
    print("Overlapping numbers:", overlap)
    print("Overlap as % of table1 risk numbers:", round(overlap_pct_of_t1 * 100, 2), "%")
    print("Overlap as % of table2 numbers:", round(overlap_pct_of_t2 * 100, 2), "%")

    print("\n=== TRAFFIC IMPACT ===")
    print("Total calls:", total_calls)
    print("Calls from risk numbers:", risk_calls)
    print("Risk call share:", round(risk_call_share * 100, 2), "%")

    print("\n=== CAUSE-16 COMPARISON ===")
    print("Cause16 rate in risk-number calls:", round(risk_cause16_share * 100, 2), "%")
    print("Cause16 rate in non-risk calls:", round(nonrisk_cause16_share * 100, 2), "%")

    # 6) Segment comparison
    print("\n=== SEGMENT COMPARISON (risk vs non-risk, duration) ===")
    seg_df = t2[["is_risk", "Duration_num"]].copy()
    seg_df["segment"] = np.where(seg_df["is_risk"], "risk_numbers", "non_risk_numbers")
    cmp_result = compare_segments(seg_df, segment_col="segment", metric_col="Duration_num")
    print(cmp_result["interpretation"])
    if cmp_result.get("summary") is not None:
        print(cmp_result["summary"].to_string(index=False))

    # 7) Concentration among risk numbers
    print("\n=== CONCENTRATION (calls among risk numbers) ===")
    risk_calls_by_num = (
        t2[t2["is_risk"]]
        .groupby("num_norm", as_index=False)
        .size()
        .rename(columns={"size": "call_count"})
    )
    conc_result = concentration_analysis(
        risk_calls_by_num, value_col="call_count", entity_col="num_norm"
    )
    print(conc_result["interpretation"])

    # 8) Score findings and synthesize narrative
    findings = [
        {
            "description": "Risk-number calls have a higher cause-16 rate vs baseline",
            "metric_value": float(risk_cause16_share),
            "baseline_value": float(nonrisk_cause16_share),
            "affected_pct": float(risk_call_share),
            "actionable": True,
            "confidence": 0.75,
            "category": "anomaly",
            "direction": "up",
            "metric_name": "cause16_rate",
        },
        {
            "description": "Risk traffic concentration indicates a small subset drives most flagged activity",
            "metric_value": float(conc_result["top_20_pct_share"]),
            "baseline_value": 0.20,
            "affected_pct": float(overlap_pct_of_t2),
            "actionable": True,
            "confidence": 0.70,
            "category": "segment",
            "direction": "up",
            "metric_name": "top20_share",
        },
    ]

    scored = score_findings(findings)
    story = synthesize_insights(
        scored["ranked_findings"],
        metadata={
            "dataset_name": "my_experiment",
            "question": "How should we prioritize spam-risk numbers in CDR?",
        },
    )

    print("\n=== PRIORITIZED FINDINGS ===")
    print(scored["interpretation"])

    print("\n=== STORY HEADLINE ===")
    print(story["headline"])

    print("\n=== NARRATIVE FLOW ===")
    for i, beat in enumerate(story["narrative_flow"], 1):
        print(str(i) + ".", beat)

    print("\n=== RECOMMENDED ACTION ITEMS ===")
    for item in story["action_items"][:5]:
        print("-", item)

if __name__ == "__main__":
    main()

