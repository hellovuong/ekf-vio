#!/usr/bin/env python3
# Copyright (c) 2026, Long Vuong
# SPDX-License-Identifier: BSD-3-Clause
"""
Regression checker — compare evaluate_euroc.py metrics.csv output against
stored baselines and exit non-zero if any threshold is exceeded.

Usage:
    python3 scripts/check_regression.py \
        --metrics  results/MH_01_easy/metrics.csv \
        --baselines test/regression/baselines.json \
        --sequence  MH_01_easy

Exit codes:
    0  all checks passed
    1  regression detected (RPE above baseline + tolerance)
    2  bad arguments / file not found
"""

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    """Read key=value CSV produced by evaluate_euroc.py save_report()."""
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "metric,value":
                continue
            key, _, value = line.partition(",")
            metrics[key.strip()] = float(value.strip())
    return metrics


def check_metric(name, value, baseline, tol, unit, fmt=".6f"):
    limit = baseline + tol
    if value > limit:
        print(f"  [FAIL] {name} {value:{fmt}} {unit} > limit {limit:{fmt}} {unit}  "
              f"(regression of {value - baseline:{fmt}} {unit} above baseline)")
        return True
    margin = limit - value
    print(f"  [PASS] {name} within limit  (margin: {margin:{fmt}} {unit})")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="EKF-VIO regression gate.")
    parser.add_argument("--metrics",   required=True, help="Path to metrics.csv")
    parser.add_argument("--baselines", required=True, help="Path to baselines.json")
    parser.add_argument("--sequence",  required=True, help="Sequence name key in baselines.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write the current RPE as the new baseline (use after intentional improvement)")
    args = parser.parse_args()

    metrics_path   = Path(args.metrics)
    baselines_path = Path(args.baselines)

    if not metrics_path.exists():
        print(f"[ERROR] metrics file not found: {metrics_path}", file=sys.stderr)
        return 2
    if not baselines_path.exists():
        print(f"[ERROR] baselines file not found: {baselines_path}", file=sys.stderr)
        return 2

    with open(baselines_path) as f:
        baselines = json.load(f)

    seq = args.sequence
    if seq not in baselines["sequences"]:
        print(f"[ERROR] sequence '{seq}' not found in baselines.json. "
              f"Available: {list(baselines['sequences'].keys())}", file=sys.stderr)
        return 2

    bl   = baselines["sequences"][seq]
    mets = load_metrics(metrics_path)

    rpe_trans = mets.get("rpe_trans_rmse")
    rpe_rot   = mets.get("rpe_rot_rmse_deg")

    if rpe_trans is None or rpe_rot is None:
        print("[ERROR] 'rpe_trans_rmse' or 'rpe_rot_rmse_deg' not found in metrics.csv — "
              "re-run evaluate_euroc.py to regenerate.", file=sys.stderr)
        return 2

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'─' * 56}")
    print(f"  Regression check: {seq}")
    print(f"{'─' * 56}")
    print(f"  RPE trans RMSE : {rpe_trans:.6f} m")
    print(f"  RPE rot   RMSE : {rpe_rot:.6f} deg")
    if "ate_rmse" in mets:
        print(f"  ATE RMSE       : {mets['ate_rmse']:.6f} m  (informational)")
    if "drift_pct" in mets:
        print(f"  Drift          : {mets['drift_pct']:.4f} %  (informational)")
    print(f"{'─' * 56}")

    failed = False

    # Check 1 — RPE translation
    failed |= check_metric(
        "RPE trans RMSE",
        rpe_trans,
        bl["rpe_trans_rmse_m"],
        bl["rpe_trans_tol_m"],
        "m",
    )

    # Check 2 — RPE rotation
    failed |= check_metric(
        "RPE rot   RMSE",
        rpe_rot,
        bl["rpe_rot_rmse_deg"],
        bl["rpe_rot_tol_deg"],
        "deg",
    )

    # Check 3 — trajectory length (catches early crashes / short runs)
    min_poses = bl.get("min_poses", 0)
    if "traj_len_gt" in mets and min_poses > 0:
        traj_len = mets["traj_len_gt"]
        if traj_len < 5.0:
            print(f"  [FAIL] GT trajectory length {traj_len:.2f} m looks too short — "
                  f"sequence may not have run fully")
            failed = True
        else:
            print(f"  [PASS] Trajectory length {traj_len:.2f} m")

    print(f"{'─' * 56}\n")

    # ── Optional: update baseline ──────────────────────────────────────────
    if args.update_baseline and not failed:
        bl["rpe_trans_rmse_m"] = round(rpe_trans, 6)
        bl["rpe_rot_rmse_deg"] = round(rpe_rot, 6)
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"[INFO] Baseline updated: {seq}  "
              f"rpe_trans_rmse_m={rpe_trans:.6f}  rpe_rot_rmse_deg={rpe_rot:.6f}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
