# EKF Visual-Inertial Odometry for VR/AR
### Snapchat — VSLAM Engineer Interview
**Long Vuong**

---

## Slide 1 — Agenda

1. Problem: Why VSLAM is Hard in VR/AR
2. System Architecture
3. State Estimation — EKF Formulation
4. Feature Tracking Pipeline
5. Performance Bottleneck Analysis & Optimisation
6. Benchmark Results
7. Open Challenges & Future Work

---

## Slide 2 — The Problem: VSLAM for VR/AR

### Why VSLAM on a Headset is Uniquely Challenging

| Constraint | Typical Camera | VR/AR Headset |
|---|---|---|
| Latency budget | 100 ms | **< 5 ms** (motion-to-photon) |
| Compute | Server / laptop | Embedded SoC (< 5 W) |
| Motion profile | Slow, controlled | Fast rotational (head shake) |
| Environment | Outdoor, structured | Indoor, textureless walls |
| Failure cost | degraded UX | **Nausea, safety issue** |

### Key Requirements
- **Real-time** at camera rate (≥ 30 Hz) on embedded hardware
- **Low ATE** (< 5 cm typical room scale) for stable hologram anchoring
- **Fast re-initialisation** after occlusion or aggressive motion
- **No global infrastructure** (no GPS, no markers)

---

## Slide 3 — System Architecture

```
IMU (~200 Hz) ──────────────────► EKF PREDICT (RK4)
                                        │
Left cam  ──► StereoRectifier ──► StereoTracker ──► 3D features
Right cam ─────────┘                                     │
                                                         ▼
                                                   EKF UPDATE
                                              (sequential Kalman,
                                               4 residuals/feature)
                                                         │
                                                         ▼
                                                 T_wb, v, b_g, b_a
```

### Sensor Fusion Philosophy
- **IMU at high rate (200 Hz)** — predicts pose between frames; absorbs fast rotations
- **Stereo at low rate (20–30 Hz)** — corrects accumulated drift; provides absolute scale
- **Tight coupling** — visual measurements directly constrain EKF state (no intermediate pose estimate)

---

## Slide 4 — State Vector

### 16-D Nominal State, 15-D Error State

| Variable | Symbol | Dim | Notes |
|---|---|---|---|
| Pose | `T_wb` (SE3) | 6 | World ← Body via Sophus |
| Velocity | `v` | 3 | World frame (ENU) |
| Gyro bias | `b_g` | 3 | Body frame, random walk |
| Accel bias | `b_a` | 3 | Body frame, random walk |
| **Covariance** | **P** | **15×15** | Error state (SO(3) tangent for rotation) |

### Why Error-State EKF?
- Quaternion has 4 DOF but only 3 physical DOF → over-parameterisation causes singularities in batch methods
- Error state Δθ ∈ ℝ³ via SO(3) exponential map keeps covariance minimal and well-conditioned
- Boxplus update: `q ← exp(Δθ) ⊗ q` — numerically stable, no quaternion renormalisation needed

---

## Slide 5 — IMU Predict Step (RK4)

### Continuous-Time Kinematics

```
ṗ = v
v̇ = R · (a_meas − b_a) + g_w          g_w = [0, 0, −9.81] m/s²
q̇ = q ⊗ [0, (ω_meas − b_g) / 2]
ḃ_g = 0,  ḃ_a = 0
```

### 4th-Order Runge-Kutta Integration
- First-order Euler would accumulate significant error at 200 Hz with aggressive head motion
- RK4 cuts integration error from O(dt²) to O(dt⁴) with only 4× IMU evaluations

### Error-State Propagation (F matrix, 15×15)

```
F = [ 0   I    0         0    0  ]   ← δṗ
    [ 0   0   −R[a_c]×   0   −R  ]   ← δv̇
    [ 0   0   −[ω_c]×   −I    0  ]   ← δθ̇
    [ 0   0    0          0    0  ]   ← δḃ_g
    [ 0   0    0          0    0  ]   ← δḃ_a

Φ = I + F·Δt
P ← Φ P Φᵀ + G Qc Gᵀ · Δt
```

### Optimisation: Precomputed `G·Qc·Gᵀ`
`G·Qc·Gᵀ` only contains `σ²·I` blocks — **rotation-independent**, constant for all time.
Precomputed once at construction → saves one 15×12 matrix build + triple product per IMU step (200 calls/sec).

---

## Slide 6 — Feature Tracking Pipeline

```
┌─────────────────────────────────────────────────────┐
│  1. FAST detection (masked around existing tracks)  │
│  2. Temporal LK:  prev_left → curr_left             │  ← Uses pre-built pyramid
│  3. F-matrix RANSAC outlier rejection               │
│  4. Stereo LK:   curr_left → curr_right             │  ← Epipolar-constrained
│  5. Disparity check: [1, 200] px                    │
│  6. Triangulate: Z = fx·baseline / disparity        │
└─────────────────────────────────────────────────────┘
Output: {id, u_l, v_l, u_r, v_r, p_c}  per feature
```

### Key Implementation Details

**Pre-built LK pyramid** — `cv::buildOpticalFlowPyramid()` is called once at the end of each frame and cached. The next frame reuses it directly, eliminating one pyramid build per frame.

**Persistent feature IDs** — features carry an integer ID across frames. The EKF can associate 3D landmarks with tracked pixels without re-triangulating each frame.

**Epipolar verification** — after stereo LK, a vertical-disparity check `|y_l − y_r| < ε` rejects matches that drifted off the rectified epipolar line.

---

## Slide 7 — EKF Visual Update

### Measurement Model (per feature)

For a 3D point `p_c = [X, Y, Z]` in the left-camera frame:

```
h(p_c) = [ fx·X/Z + cx,          ]   ← u_l
          [ fy·Y/Z + cy,          ]   ← v_l
          [ fx·(X−b)/Z + cx,      ]   ← u_r   (b = stereo baseline)
          [ fy·Y/Z + cy           ]   ← v_r   (= v_l for rectified stereo)
```

**Jacobian H** (4×15): chain rule through the projection, the rigid body transform `T_wc`, and the EKF state.

**Residual**: `z = z_meas − h(p_c)` — gated by Mahalanobis distance and a 40-px hard pixel limit.

---

## Slide 8 — The Performance Bottleneck Discovery

### Initial Profile (V1_01_easy, 2911 frames)

| Stage | ms/frame | Share |
|---|---|---|
| Stereo rectification | 0.55 | 2 % |
| Feature tracking | 1.44 | 5 % |
| **EKF update** | **25.46** | **93 %** |
| Total | 27.45 | — |
| **Throughput** | **36 Hz** | — |

### Root Cause: O((4N)³) LDLT on a 800×800 Matrix

The batch EKF update assembled all N features into one stacked measurement:
```
S = H_all · P · H_all^T + R_mat       // (4N × 4N) matrix, N ≈ 200
```
Solved via `Eigen::LDLT<MatrixXd>(S)` — **~512M FLOP per camera frame** at N = 200.

At 20 Hz stereo, this single matrix solve consumed the entire CPU budget.

---

## Slide 9 — The Fix: Sequential Kalman Update

### Mathematical Equivalence
For independent measurement noise (each feature's reprojection error is independent), the batch and sequential Kalman updates produce **identical results** in the linear case.

### Sequential Loop (N features → N × 4×4 solves)

```
for each feature k:
    S_k = H_k · P · H_k^T + R_i          // 4×4 Cholesky — trivially cheap
    K_k = P · H_k^T · S_k⁻¹             // 15×4
    // Apply correction immediately (keeps P and state consistent)
    state ⊞ K_k · residual_k
    IKH  = I − K_k · H_k
    P    = IKH · P · IKH^T + K_k · R_i · K_k^T   // Joseph form
    P    = 0.5 · (P + Pᵀ)                          // symmetrise
```

**Cost per feature**: ~2 matrix products of size 15×15 ≈ **10,000 FLOP**
**Total for N=200**: ~**2M FLOP** vs 512M → **~250× reduction in arithmetic**

### Why Sequential is Also More Accurate
- After each feature, `P` is tightened before computing `K_{k+1}` → later redundant features get smaller gains automatically (implicit down-weighting)
- Each 4×4 Cholesky is always well-conditioned; the 800×800 batch matrix was numerically ill-conditioned with near-collinear features
- Each Jacobian is evaluated against a progressively-corrected state → less nonlinear approximation error

---

## Slide 10 — Performance After Optimisation

### Measured on EuRoC V1_01_easy (2911 frames, single thread, RelWithDebInfo)

| Stage | Before | After | Speedup |
|---|---|---|---|
| EKF update | 25.46 ms | **0.26 ms** | **98×** |
| Total pipeline | 27.45 ms | **1.96 ms** | **14×** |
| Throughput | 36 Hz | **510 Hz** | **14×** |

### Accuracy — ATE (Umeyama-aligned)

```
GT  trajectory length :  58.56 m
Est trajectory length :  58.82 m

ATE RMSE   :  0.322 m      (0.55% of trajectory length)
RPE Trans  :  0.042 m/frame
RPE Rot    :  0.17 deg/frame
```

**0.55% drift** is competitive with published monocular VIO systems on EuRoC.

---

## Slide 11 — Relevance to VR/AR at Snapchat

### Mapping the Work to Spectacles / AR Headset Requirements

| This System | AR Headset Requirement | Gap / Next Step |
|---|---|---|
| 510 Hz VIO pipeline | < 5 ms motion-to-photon | ✅ Budget headroom for rendering |
| Sequential Kalman | Fixed-time-per-feature update | ✅ Real-time-safe, no latency spikes |
| Stereo + IMU fusion | 6-DOF tracking without GPS | ✅ Indoor/outdoor works |
| 0.55% drift per 58m | Hologram stability | ⚠ Need loop closure for long sessions |
| 15D error-state EKF | Compact state for embedded | ✅ < 1 MB working set |
| FAST + LK tracking | Low-power feature front-end | ✅ Can port to SIMD/GPU |

### What VR/AR Adds on Top
- **Rolling-shutter correction** — phone/headset cameras have per-row time offset
- **Photometric calibration** — consistent brightness for better LK tracking under varying lighting
- **SLAM / loop closure** — pose graph backend (g2o / GTSAM) for room-scale drift correction
- **Multi-camera** — wide-baseline or fisheye camera models for larger FOV tracking
- **On-device ML** — depth estimation or feature learning to replace hand-crafted FAST/LK in textureless scenes

---

## Slide 12 — Open Challenges & Active Work

### Current Diagnostic: Stereo Match Rate

On V1_01_easy, profiling revealed that LK stereo matching converges to wrong local minima for ~93% of candidate features:

```
stereoMatch: lk_ok=286/300  epipolar_fail=265  final_ok=21
```

**Root cause**: LK initialized at `right_pts = left_pts` (zero-disparity guess) converges to nearby texture patches off the epipolar line.

**Fix in progress**: For tracked features, initialise the stereo LK search from `prev_right_pt + left_motion_delta` — a near-correct starting point that avoids the wrong local minimum.

### Broader Roadmap

| Priority | Item |
|---|---|
| 🔴 High | Better stereo initialisation (prev-right + motion delta) |
| 🔴 High | Loop closure / relocalization (bag-of-words or NetVLAD) |
| 🟡 Medium | MSCKF null-space projection (reduce linearisation error) |
| 🟡 Medium | Static initialiser (gravity alignment from first N IMU readings) |
| 🟢 Low | Rolling-shutter correction |
| 🟢 Low | Learned feature front-end (SuperPoint / LightGlue) |

---

## Slide 13 — Code Quality & Engineering Practice

### What the Codebase Demonstrates

| Practice | Detail |
|---|---|
| **Modern C++20** | `std::ranges`, concepts, structured bindings |
| **Lie groups (Sophus)** | SE3 / SO3 operations; avoids gimbal lock, correct manifold updates |
| **Profiling before optimising** | Measured per-stage timers — found EKF, not LK, was the bottleneck |
| **Test coverage (GTest)** | Unit tests for EKF predict/update, tracker, VO |
| **CI / Docker** | GitHub Actions runs build + test + lint on every PR |
| **Pre-commit hooks** | clang-format, gitleaks, SPDX license headers |
| **Async visualisation** | Producer/consumer queue for Rerun I/O — never stalls the EKF loop |

### Repository
`github.com/hellovuong/ekf-vio`  — BSD-3-Clause

---

## Slide 14 — Key Takeaways

> **The bottleneck is always where you don't expect it.**
> Profiling showed the visual feature tracker (assumed to be slow) took 1.4 ms/frame,
> while the EKF matrix solve (assumed to be cheap) took 25 ms/frame.
> A single algorithmic change (sequential vs. batch update) delivered a **98× speedup**
> with **no loss in accuracy**.

### For Snapchat / Spectacles

- A 510 Hz VIO pipeline leaves **ample headroom** for pose prediction, rendering, and reprojection on embedded hardware
- Sequential Kalman gives **bounded, predictable latency** — important for smooth hologram rendering
- The codebase is structured for extension: MSCKF, loop closure, and learned features can be added as separate modules

---

## Appendix A — Build & Run

```bash
# Build
colcon build --cmake-args -DWITH_RERUN=ON

# Run EKF-VIO on EuRoC
./install/ekf_vio/lib/ekf_vio/euroc_runner \
    /data/V1_01_easy config/euroc.yaml

# Run with Rerun visualisation
./install/ekf_vio/lib/ekf_vio/euroc_runner \
    /data/V1_01_easy config/euroc.yaml --rerun

# Evaluate trajectory
python3 scripts/evaluate_euroc.py \
    --est euroc_traj.csv \
    --gt /data/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv
```

---

## Appendix B — Key References

- Mourikis & Roumeliotis, **MSCKF**, ICRA 2007 — sequential vs. batch Kalman for visual features
- Forster et al., **SVO**, ICRA 2014 — semi-direct visual odometry
- Qin, Li, Shen, **VINS-Mono**, T-RO 2018 — tightly-coupled monocular VIO
- Burri et al., **EuRoC MAV Dataset**, IJRR 2016 — benchmark dataset used here
- Chirikjian, **Stochastic Models in Engineering**, 2011 — error-state Kalman on Lie groups
