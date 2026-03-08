#!/usr/bin/env python3
# Copyright (c) 2026, Long Vuong
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluate EKF-VIO trajectory against EuRoC ground truth.

Usage:
    python3 scripts/evaluate_euroc.py \
        --est  euroc_traj.csv \
        --gt   /path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
        --out  results/MH_01

Metrics computed:
    - ATE  (Absolute Trajectory Error)   : RMSE, mean, median, max, std
    - RPE  (Relative Pose Error)         : translation & rotation at fixed Δt
    - Drift                              : final position / total distance
    - Trajectory plots                   : 3D, XY, XZ, error-vs-time
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rotation helpers (avoid scipy dependency for minimal installs)
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def rot_to_angle(R):
    """Rotation matrix -> angle (radians) via trace."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_estimated(path):
    """Load estimated trajectory CSV: timestamp,px,py,pz,qw,qx,qy,qz"""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            vals = [float(v) for v in parts[:8]]
            data.append(vals)
    data = np.array(data)
    return {
        "t": data[:, 0],
        "p": data[:, 1:4],
        "q": data[:, 4:8],   # [qw, qx, qy, qz]
    }


def load_euroc_gt(path):
    """Load EuRoC ground truth CSV.
    Format: timestamp[ns], p_x,p_y,p_z, q_w,q_x,q_y,q_z, v_x,v_y,v_z,
            bw_x,bw_y,bw_z, ba_x,ba_y,ba_z
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 17:
                continue
            vals = [float(v) for v in parts[:17]]
            data.append(vals)
    data = np.array(data)
    t_sec = data[:, 0] * 1e-9
    return {
        "t": t_sec,
        "p": data[:, 1:4],
        "q": data[:, 4:8],   # [qw, qx, qy, qz]
        "v": data[:, 8:11],
    }


# ---------------------------------------------------------------------------
# Time association
# ---------------------------------------------------------------------------

def associate(t_est, t_gt, max_dt=0.02):
    """Associate estimated timestamps to nearest GT timestamps.
    Returns arrays of matched indices (est_idx, gt_idx).
    """
    est_idx, gt_idx = [], []
    j = 0
    for i, te in enumerate(t_est):
        while j < len(t_gt) - 1 and t_gt[j + 1] <= te:
            j += 1
        # check j and j+1
        best_j = j
        if j + 1 < len(t_gt) and abs(t_gt[j + 1] - te) < abs(t_gt[j] - te):
            best_j = j + 1
        if abs(t_gt[best_j] - te) <= max_dt:
            est_idx.append(i)
            gt_idx.append(best_j)
    return np.array(est_idx), np.array(gt_idx)


# ---------------------------------------------------------------------------
# SE(3) alignment (Umeyama)
# ---------------------------------------------------------------------------

def align_umeyama(p_est, p_gt, with_scale=False):
    """Compute SE(3) alignment (R, t, s) such that  s*R*p_est + t ≈ p_gt.
    Returns (R, t, s).  If with_scale=False, s=1.
    """
    mu_est = p_est.mean(axis=0)
    mu_gt  = p_gt.mean(axis=0)

    est_c = p_est - mu_est
    gt_c  = p_gt  - mu_gt

    H = est_c.T @ gt_c  # 3x3

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T

    if with_scale:
        var_est = np.sum(est_c ** 2)
        s = np.sum(S * np.diag(D)) / var_est
    else:
        s = 1.0

    t = mu_gt - s * R @ mu_est
    return R, t, s


def apply_alignment(p, R, t, s):
    return s * (R @ p.T).T + t


# ---------------------------------------------------------------------------
# ATE (Absolute Trajectory Error)
# ---------------------------------------------------------------------------

def compute_ate(p_est_aligned, p_gt):
    """Returns per-frame translational errors (Euclidean)."""
    diff = p_est_aligned - p_gt
    return np.linalg.norm(diff, axis=1)


def ate_stats(errors):
    return {
        "rmse":   np.sqrt(np.mean(errors ** 2)),
        "mean":   np.mean(errors),
        "median": np.median(errors),
        "std":    np.std(errors),
        "max":    np.max(errors),
        "min":    np.min(errors),
    }


# ---------------------------------------------------------------------------
# RPE (Relative Pose Error)
# ---------------------------------------------------------------------------

def compute_rpe(p_est, q_est, p_gt, q_gt, delta_frames=10):
    """Compute relative pose error at fixed frame intervals.
    Returns (trans_errors, rot_errors_deg).
    """
    n = len(p_est)
    trans_errs = []
    rot_errs = []

    for i in range(n - delta_frames):
        j = i + delta_frames

        # Estimated relative transform
        R_ei = quat_to_rot(q_est[i])
        R_ej = quat_to_rot(q_est[j])
        dp_est = R_ei.T @ (p_est[j] - p_est[i])
        dR_est = R_ei.T @ R_ej

        # GT relative transform
        R_gi = quat_to_rot(q_gt[i])
        R_gj = quat_to_rot(q_gt[j])
        dp_gt = R_gi.T @ (p_gt[j] - p_gt[i])
        dR_gt = R_gi.T @ R_gj

        # Error
        dp_err = dp_est - dp_gt
        dR_err = dR_est @ dR_gt.T

        trans_errs.append(np.linalg.norm(dp_err))
        rot_errs.append(np.degrees(rot_to_angle(dR_err)))

    return np.array(trans_errs), np.array(rot_errs)


# ---------------------------------------------------------------------------
# Trajectory length & drift
# ---------------------------------------------------------------------------

def trajectory_length(p):
    diffs = np.diff(p, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(t_est, p_est_al, p_gt, ate_errs, rpe_t, rpe_r, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ---- 3D trajectory ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
            "b-", linewidth=1.2, label="Ground Truth")
    ax.plot(p_est_al[:, 0], p_est_al[:, 1], p_est_al[:, 2],
            "r-", linewidth=1.0, alpha=0.8, label="Estimated (aligned)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title("3D Trajectory")
    fig.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- XY top-down ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(p_gt[:, 0], p_gt[:, 1], "b-", linewidth=1.2, label="Ground Truth")
    ax.plot(p_est_al[:, 0], p_est_al[:, 1], "r-", linewidth=1.0,
            alpha=0.8, label="Estimated")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.legend()
    ax.set_title("Top-Down (XY)")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "trajectory_xy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- XZ side view ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p_gt[:, 0], p_gt[:, 2], "b-", linewidth=1.2, label="Ground Truth")
    ax.plot(p_est_al[:, 0], p_est_al[:, 2], "r-", linewidth=1.0,
            alpha=0.8, label="Estimated")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.legend()
    ax.set_title("Side View (XZ)")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "trajectory_xz.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- ATE vs time ----
    fig, ax = plt.subplots(figsize=(10, 4))
    t_plot = t_est - t_est[0]
    ax.plot(t_plot, ate_errs, "r-", linewidth=0.8)
    ax.axhline(np.mean(ate_errs), color="k", linestyle="--", linewidth=0.8,
               label=f"Mean = {np.mean(ate_errs):.4f} m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ATE [m]")
    ax.set_title("Absolute Trajectory Error over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "ate_vs_time.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- ATE histogram ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ate_errs, bins=50, color="steelblue", edgecolor="white",
            alpha=0.85)
    ax.axvline(np.mean(ate_errs), color="r", linestyle="--",
               label=f"Mean = {np.mean(ate_errs):.4f} m")
    ax.axvline(np.median(ate_errs), color="orange", linestyle="--",
               label=f"Median = {np.median(ate_errs):.4f} m")
    ax.set_xlabel("ATE [m]")
    ax.set_ylabel("Count")
    ax.set_title("ATE Distribution")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "ate_histogram.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- RPE translation & rotation ----
    if len(rpe_t) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        rpe_x = np.arange(len(rpe_t))

        ax1.plot(rpe_x, rpe_t, "r-", linewidth=0.6)
        ax1.set_ylabel("RPE trans [m]")
        ax1.set_title("Relative Pose Error")
        ax1.grid(True, alpha=0.3)

        ax2.plot(rpe_x, rpe_r, "b-", linewidth=0.6)
        ax2.set_xlabel("Frame pair index")
        ax2.set_ylabel("RPE rot [deg]")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rpe.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- Per-axis position error ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        err_i = p_est_al[:, i] - p_gt[:, i]
        ax.plot(t_plot, err_i, linewidth=0.8)
        ax.set_ylabel(f"Δ{label} [m]")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
    axes[0].set_title("Per-Axis Position Error (aligned)")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_axis_error.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(stats, rpe_t, rpe_r, traj_len_gt, traj_len_est, scale):
    sep = "-" * 52
    print(f"\n{'=' * 52}")
    print("  EKF-VIO Evaluation Report")
    print(f"{'=' * 52}")

    print(f"\n{'Trajectory Statistics':^52}")
    print(sep)
    print(f"  GT  trajectory length : {traj_len_gt:10.4f} m")
    print(f"  Est trajectory length : {traj_len_est:10.4f} m")
    print(f"  Scale factor (Umeyama): {scale:10.6f}")

    print(f"\n{'ATE — Absolute Trajectory Error [m]':^52}")
    print(sep)
    print(f"  RMSE   : {stats['rmse']:10.6f}")
    print(f"  Mean   : {stats['mean']:10.6f}")
    print(f"  Median : {stats['median']:10.6f}")
    print(f"  Std    : {stats['std']:10.6f}")
    print(f"  Min    : {stats['min']:10.6f}")
    print(f"  Max    : {stats['max']:10.6f}")

    if len(rpe_t) > 0:
        print(f"\n{'RPE — Relative Pose Error':^52}")
        print(sep)
        print(f"  Trans RMSE  : {np.sqrt(np.mean(rpe_t ** 2)):10.6f} m")
        print(f"  Trans Mean  : {np.mean(rpe_t):10.6f} m")
        print(f"  Trans Median: {np.median(rpe_t):10.6f} m")
        print(f"  Rot   RMSE  : {np.sqrt(np.mean(rpe_r ** 2)):10.6f} deg")
        print(f"  Rot   Mean  : {np.mean(rpe_r):10.6f} deg")
        print(f"  Rot   Median: {np.median(rpe_r):10.6f} deg")

    drift = stats["rmse"] / traj_len_gt * 100.0 if traj_len_gt > 0 else float("inf")
    print(f"\n{'Drift':^52}")
    print(sep)
    print(f"  ATE RMSE / GT length : {drift:10.4f} %")
    print(f"{'=' * 52}\n")


def save_report(stats, rpe_t, rpe_r, traj_len_gt, traj_len_est, scale,
                out_path):
    """Save metrics to a CSV file for batch comparisons."""
    with open(out_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"ate_rmse,{stats['rmse']:.8f}\n")
        f.write(f"ate_mean,{stats['mean']:.8f}\n")
        f.write(f"ate_median,{stats['median']:.8f}\n")
        f.write(f"ate_std,{stats['std']:.8f}\n")
        f.write(f"ate_min,{stats['min']:.8f}\n")
        f.write(f"ate_max,{stats['max']:.8f}\n")
        if len(rpe_t) > 0:
            f.write(f"rpe_trans_rmse,{np.sqrt(np.mean(rpe_t**2)):.8f}\n")
            f.write(f"rpe_trans_mean,{np.mean(rpe_t):.8f}\n")
            f.write(f"rpe_rot_rmse_deg,{np.sqrt(np.mean(rpe_r**2)):.8f}\n")
            f.write(f"rpe_rot_mean_deg,{np.mean(rpe_r):.8f}\n")
        f.write(f"traj_len_gt,{traj_len_gt:.8f}\n")
        f.write(f"traj_len_est,{traj_len_est:.8f}\n")
        f.write(f"scale,{scale:.8f}\n")
        drift = stats["rmse"] / traj_len_gt * 100 if traj_len_gt > 0 else 0
        f.write(f"drift_pct,{drift:.8f}\n")
    print(f"[INFO] Metrics saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EKF-VIO trajectory against EuRoC ground truth.")

    parser.add_argument("--est", required=True,
                        help="Estimated trajectory CSV "
                             "(timestamp,px,py,pz,qw,qx,qy,qz)")
    parser.add_argument("--gt", required=True,
                        help="EuRoC ground truth CSV "
                             "(state_groundtruth_estimate0/data.csv)")
    parser.add_argument("--out", default="results",
                        help="Output directory for plots and metrics "
                             "(default: results)")
    parser.add_argument("--align-scale", action="store_true",
                        help="Align with scale (Sim(3)) — use for monocular")
    parser.add_argument("--max-dt", type=float, default=0.02,
                        help="Max time difference for association [s] "
                             "(default: 0.02)")
    parser.add_argument("--rpe-delta", type=int, default=10,
                        help="Frame delta for RPE computation (default: 10)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")

    args = parser.parse_args()

    # Load data
    print(f"[INFO] Loading estimated trajectory: {args.est}")
    est = load_estimated(args.est)
    print(f"       -> {len(est['t'])} poses")

    print(f"[INFO] Loading ground truth: {args.gt}")
    gt = load_euroc_gt(args.gt)
    print(f"       -> {len(gt['t'])} poses")

    # Associate
    ei, gi = associate(est["t"], gt["t"], max_dt=args.max_dt)
    print(f"[INFO] Associated {len(ei)} / {len(est['t'])} estimated poses "
          f"(max Δt = {args.max_dt}s)")

    if len(ei) < 3:
        print("[ERROR] Too few associations — check timestamps.", file=sys.stderr)
        sys.exit(1)

    p_est = est["p"][ei]
    q_est = est["q"][ei]
    t_est = est["t"][ei]
    p_gt  = gt["p"][gi]
    q_gt  = gt["q"][gi]

    # Align
    R_al, t_al, s_al = align_umeyama(p_est, p_gt, with_scale=args.align_scale)
    p_est_al = apply_alignment(p_est, R_al, t_al, s_al)

    # ATE
    ate_errs = compute_ate(p_est_al, p_gt)
    stats = ate_stats(ate_errs)

    # RPE
    rpe_t, rpe_r = compute_rpe(p_est_al, q_est, p_gt, q_gt,
                               delta_frames=args.rpe_delta)

    # Trajectory lengths
    traj_len_gt  = trajectory_length(p_gt)
    traj_len_est = trajectory_length(p_est_al)

    # Report
    print_report(stats, rpe_t, rpe_r, traj_len_gt, traj_len_est, s_al)

    # Save
    os.makedirs(args.out, exist_ok=True)
    save_report(stats, rpe_t, rpe_r, traj_len_gt, traj_len_est, s_al,
                os.path.join(args.out, "metrics.csv"))

    # Save aligned trajectory for external tools (e.g. evo)
    aligned_path = os.path.join(args.out, "est_aligned.csv")
    with open(aligned_path, "w") as f:
        f.write("# timestamp,px,py,pz,qw,qx,qy,qz\n")
        for k in range(len(t_est)):
            f.write(f"{t_est[k]:.9f},"
                    f"{p_est_al[k,0]:.9f},{p_est_al[k,1]:.9f},"
                    f"{p_est_al[k,2]:.9f},"
                    f"{q_est[k,0]:.9f},{q_est[k,1]:.9f},"
                    f"{q_est[k,2]:.9f},{q_est[k,3]:.9f}\n")
    print(f"[INFO] Aligned trajectory saved to {aligned_path}")

    # Plots
    if not args.no_plot:
        make_plots(t_est, p_est_al, p_gt, ate_errs, rpe_t, rpe_r, args.out)


if __name__ == "__main__":
    main()
