#!/usr/bin/env python3
"""Compare VO-only vs EKF-VIO trajectories against EuRoC ground truth.

Metrics:
  - ATE  (Absolute Trajectory Error): Umeyama-aligned RMSE of positions
  - RPE  (Relative Pose Error): translation/rotation drift per segment
  - Final position error
  - Trajectory length

Usage:
  python3 compare_vo_vio.py <vo_traj.csv> <vio_traj.csv> <gt_csv>
"""

import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_traj(path: str) -> np.ndarray:
    """Load trajectory CSV: timestamp,px,py,pz,qw,qx,qy,qz"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split(",")]
            rows.append(vals[:8])  # t, px, py, pz, qw, qx, qy, qz
    return np.array(rows)


def load_euroc_gt(path: str) -> np.ndarray:
    """Load EuRoC ground truth CSV (nanosecond timestamps → seconds)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split(",")]
            t_sec = vals[0] * 1e-9
            # px,py,pz, qw,qx,qy,qz
            rows.append([t_sec, vals[1], vals[2], vals[3],
                         vals[4], vals[5], vals[6], vals[7]])
    return np.array(rows)


def associate(traj: np.ndarray, gt: np.ndarray, max_dt: float = 0.02):
    """Associate trajectory timestamps to closest GT timestamps."""
    t_traj = traj[:, 0]
    t_gt = gt[:, 0]
    idx_traj, idx_gt = [], []
    for i, t in enumerate(t_traj):
        j = np.argmin(np.abs(t_gt - t))
        if abs(t_gt[j] - t) < max_dt:
            idx_traj.append(i)
            idx_gt.append(j)
    return np.array(idx_traj), np.array(idx_gt)


# ---------------------------------------------------------------------------
# Umeyama alignment (SE(3), with scale)
# ---------------------------------------------------------------------------

def umeyama_alignment(model: np.ndarray, data: np.ndarray, with_scale: bool = False):
    """Align model to data using Umeyama method. Returns s, R, t."""
    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)
    sigma2 = np.mean(np.sum((data - mu_d) ** 2, axis=1))
    H = (data - mu_d).T @ (model - mu_m) / len(model)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma2 if with_scale else 1.0
    t = mu_d - s * R @ mu_m
    return s, R, t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]
    ])


def trajectory_length(positions: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def compute_ate(traj_pos: np.ndarray, gt_pos: np.ndarray, with_scale: bool = False):
    """ATE after Umeyama alignment. Returns RMSE, mean, median, std, max."""
    s, R, t = umeyama_alignment(traj_pos, gt_pos, with_scale=with_scale)
    aligned = (s * (R @ traj_pos.T).T) + t
    errors = np.linalg.norm(aligned - gt_pos, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "max": float(np.max(errors)),
        "scale": float(s),
    }


def compute_rpe(traj: np.ndarray, gt: np.ndarray, delta: int = 10):
    """RPE over fixed-frame intervals. Returns trans and rot RMSE."""
    trans_errors, rot_errors = [], []
    for i in range(len(traj) - delta):
        j = i + delta
        # Estimated relative transform
        p_ei, q_ei = traj[i, 1:4], traj[i, 4:8]
        p_ej, q_ej = traj[j, 1:4], traj[j, 4:8]
        R_ei, R_ej = quat_to_rot(q_ei), quat_to_rot(q_ej)
        dp_e = R_ei.T @ (p_ej - p_ei)
        dR_e = R_ei.T @ R_ej

        # GT relative transform
        p_gi, q_gi = gt[i, 1:4], gt[i, 4:8]
        p_gj, q_gj = gt[j, 1:4], gt[j, 4:8]
        R_gi, R_gj = quat_to_rot(q_gi), quat_to_rot(q_gj)
        dp_g = R_gi.T @ (p_gj - p_gi)
        dR_g = R_gi.T @ R_gj

        # Error
        trans_errors.append(np.linalg.norm(dp_e - dp_g))
        dR_err = dR_g.T @ dR_e
        angle = np.arccos(np.clip((np.trace(dR_err) - 1) / 2, -1, 1))
        rot_errors.append(angle)

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "trans_mean": float(np.mean(trans_errors)),
        "rot_rmse_deg": float(np.degrees(np.sqrt(np.mean(rot_errors ** 2)))),
        "rot_mean_deg": float(np.degrees(np.mean(rot_errors))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(name: str, traj: np.ndarray, gt_full: np.ndarray):
    idx_t, idx_g = associate(traj, gt_full)
    if len(idx_t) < 10:
        print(f"  [!] Only {len(idx_t)} associated poses — skipping")
        return None

    traj_assoc = traj[idx_t]
    gt_assoc = gt_full[idx_g]

    ate = compute_ate(traj_assoc[:, 1:4], gt_assoc[:, 1:4], with_scale=(name == "VO"))
    rpe = compute_rpe(traj_assoc, gt_assoc, delta=10)
    tlen = trajectory_length(traj_assoc[:, 1:4])
    gt_len = trajectory_length(gt_assoc[:, 1:4])

    return {
        "name": name,
        "num_poses": len(idx_t),
        "traj_length": tlen,
        "gt_length": gt_len,
        "ate": ate,
        "rpe": rpe,
        "final_pos_err": float(np.linalg.norm(traj_assoc[-1, 1:4] - gt_assoc[-1, 1:4])),
    }


def print_results(results: list):
    header = f"{'Metric':<32}"
    for r in results:
        header += f"  {r['name']:>12}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    rows = [
        ("Num associated poses",     [f"{r['num_poses']}" for r in results]),
        ("Traj length (m)",          [f"{r['traj_length']:.2f}" for r in results]),
        ("GT traj length (m)",       [f"{r['gt_length']:.2f}" for r in results]),
        ("",                         [""]*len(results)),
        ("ATE RMSE (m)",             [f"{r['ate']['rmse']:.4f}" for r in results]),
        ("ATE mean (m)",             [f"{r['ate']['mean']:.4f}" for r in results]),
        ("ATE median (m)",           [f"{r['ate']['median']:.4f}" for r in results]),
        ("ATE std (m)",              [f"{r['ate']['std']:.4f}" for r in results]),
        ("ATE max (m)",              [f"{r['ate']['max']:.4f}" for r in results]),
        ("Scale (Umeyama)",          [f"{r['ate']['scale']:.4f}" for r in results]),
        ("",                         [""]*len(results)),
        ("RPE trans RMSE (m)",       [f"{r['rpe']['trans_rmse']:.4f}" for r in results]),
        ("RPE trans mean (m)",       [f"{r['rpe']['trans_mean']:.4f}" for r in results]),
        ("RPE rot RMSE (deg)",       [f"{r['rpe']['rot_rmse_deg']:.2f}" for r in results]),
        ("RPE rot mean (deg)",       [f"{r['rpe']['rot_mean_deg']:.2f}" for r in results]),
        ("",                         [""]*len(results)),
        ("Final position error (m)", [f"{r['final_pos_err']:.4f}" for r in results]),
    ]

    for label, vals in rows:
        line = f"{label:<32}"
        for v in vals:
            line += f"  {v:>12}"
        print(line)

    print("=" * len(header))


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <traj1.csv> <traj2.csv> [traj3.csv ...] <gt.csv>")
        print(f"  Last argument is always the ground truth CSV.")
        print(f"  Trajectory names are derived from filenames.")
        sys.exit(1)

    gt = load_euroc_gt(sys.argv[-1])
    trajectories = []
    for path in sys.argv[1:-1]:
        name = Path(path).stem.replace("euroc_", "").replace("_traj", "")
        traj = load_traj(path)
        trajectories.append((name.upper(), traj))
        print(f"Loaded {name.upper()}: {len(traj)} poses")
    print(f"Loaded GT: {len(gt)} poses\n")

    results = []
    for name, traj in trajectories:
        r = evaluate(name, traj, gt)
        if r:
            results.append(r)

    if results:
        print()
        print_results(results)


if __name__ == "__main__":
    main()
