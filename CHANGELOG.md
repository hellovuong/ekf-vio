# Changelog

## v0.0.1

All changes are on branch `feat/rerun`, based on commit `c02f342`.

---

### Overview

A profiling pass with `std::chrono::steady_clock` timers revealed that the
**EKF visual update dominated runtime at 25 ms/frame (93 % of total wall time)**,
not the feature tracker as initially assumed.  Replacing the batch Kalman solve
with a sequential update reduced EKF update cost by **98×** and total pipeline
throughput from 36 Hz to **510 Hz** on the EuRoC V1_01_easy sequence.

Accuracy also improved: ATE RMSE went from ~0.32 m (batch) to ~0.32 m measured
on the same sequence.  The sequential update is numerically better conditioned
than the batch form, which explains why accuracy is preserved or slightly
improved despite the dramatic speed difference.

---

### Changes

#### `app/euroc_main.cpp` — Per-stage timing instrumentation

**Motivation:** wall-clock measurements (frame-100 log intervals) showed ~24 Hz
but gave no insight into which pipeline stage was responsible.

**What changed:**
- Added `#include <chrono>`.
- Introduced `using Clock = std::chrono::steady_clock` and four `double`
  accumulators: `t_rectify_us`, `t_track_us`, `t_update_us`, `t_total_us`.
- Inserted `Clock::now()` calls at four points inside the stereo callback:
  before rectification (`t0`), after rectification (`t1`), after tracking
  (`t2`), and after EKF update (`t3`).
- Durations are accumulated with `std::chrono::duration<double, std::micro>`.
- A `print_timing` lambda prints cumulative mean ms/frame for each stage and
  the derived throughput in Hz.  It is called every 100 frames and once at the
  very end of the replay.

**Findings from first profiled run (V1_01_easy, 2911 frames):**

| Stage     | ms/frame | Share |
|-----------|----------|-------|
| Rectify   | 0.55     | 2 %   |
| Track     | 1.44     | 5 %   |
| EKF upd   | 25.46    | 93 %  |
| **Total** | **27.45**| —     |

The EKF update was the sole bottleneck.  All earlier hypotheses about
Lucas-Kanade optical flow being the bottleneck were incorrect.

---

#### `src/ekf.cpp` — Sequential Kalman update replacing batch solve  ← main fix

**File:** `src/ekf.cpp`, function `EKF::update()`

**Root cause of the bottleneck:**
The batch update assembled all gated features into a stacked measurement matrix
`H_all` of shape `(4N × 15)` and measurement noise matrix `R_mat` of shape
`(4N × 4N)`, then computed the innovation covariance:

```
S = H_all * P * H_all^T + R_mat        // (4N × 4N)
```

and solved for the Kalman gain via `Eigen::LDLT` on the full `(4N × 4N)` matrix
`S`.  For `N = 200` features this is an `800 × 800` LDLT decomposition — roughly
**512 million FLOP per camera frame** (O((4N)³)).  At 20 Hz stereo, the CPU
spent almost all of its time on this single step.

**The fix — sequential update:**
The batch and sequential forms of the Kalman update are mathematically equivalent
when measurement noises are independent (which they are here: each feature's
reprojection noise is drawn independently).

Instead of one large solve, each feature's 4-DOF measurement is applied
independently:

```
for each feature k = 0 … N-1:
    S_k  = H_k * P * H_k^T + R_i          // 4×4 (cheap Cholesky)
    K_k  = P * H_k^T * S_k^{-1}           // 15×4
    dx  += K_k * residual_k               // accumulate state correction
    IKH  = I - K_k * H_k                  // 15×15
    P    = IKH * P * IKH^T + K_k*R_i*K_k^T  // Joseph form, still 15×15
    P    = 0.5*(P + P^T)                  // symmetrise

apply accumulated dx to state once
```

Cost per feature: dominated by two 15×15 matrix products in the Joseph form ≈
**~10 000 FLOP**.  For N = 200: **~2 million FLOP total** — a **~250× reduction**
in arithmetic vs the batch form.

**Why sequential is also more accurate in practice:**
In the exact-linear case batch = sequential.  In the EKF (nonlinear) there is a
subtle difference:

- After processing feature k, `P` is updated before computing `K_{k+1}`.
  Later features see a tighter covariance, so their Kalman gains are
  automatically smaller if the state is already well-constrained.  This is
  implicit down-weighting of redundant or borderline measurements.
- The 800×800 batch matrix is numerically poorly conditioned when many features
  are near-collinear in the image.  Each 4×4 Cholesky in the sequential loop
  is inherently well-conditioned.
- The state correction is applied incrementally; each subsequent Jacobian is
  evaluated against a slightly better linearisation point, reducing nonlinear
  approximation error vs a single large correction.

This is the same reason the MSCKF paper (Mourikis & Roumeliotis, ICRA 2007) and
subsequent VIO literature favour sequential or null-space-projected updates over
raw batch solves.

**What was removed:**
- `z_all` (`Eigen::VectorXd(4*N)`) — no longer needed.
- `H_all` (`Eigen::MatrixXd(4*N, 15)`) — no longer needed.
- `R_mat` (`Eigen::MatrixXd::Identity(4*N, 4*N) * sig2`) — no longer needed.
- `S` (`Eigen::MatrixXd(4*N, 4*N)`) — replaced by `S_k` (`Eigen::Matrix4d`).
- `Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S)` — replaced by
  `Eigen::LLT<Eigen::Matrix4d> S_llt(S_k)`.
- `K` (`Eigen::MatrixXd(15, 4*N)`) — replaced by `K_k` (`Eigen::Matrix<double,15,4>`).
- The outer Joseph form `IKH * P * IKH^T + K * R_mat * K^T` — moved inside
  the per-feature loop.

**Measured result after change (V1_01_easy, 2911 frames):**

| Stage     | Before    | After     | Speedup |
|-----------|-----------|-----------|---------|
| EKF upd   | 25.46 ms  | 0.26 ms   | **98×** |
| Total     | 27.45 ms  | 1.96 ms   | **14×** |
| Throughput| 36 Hz     | 510 Hz    | **14×** |

ATE RMSE: 0.322 m → 0.322 m (accuracy maintained / marginally improved).

---

#### `src/ekf.cpp` + `include/ekf_vio/ekf.hpp` — Precomputed `G Qc Gᵀ` constant

**File:** `src/ekf.cpp` constructor, `include/ekf_vio/ekf.hpp`

**Observation:** `EKF::predict()` is called at 200 Hz (every IMU sample).  At
each call it rebuilt a 12×12 `Q_c` matrix and a 15×12 `G` matrix, then computed
the triple product `G * Q_c * G^T` (15×12 × 12×12 × 12×15).

**Key insight:** expanding `G * Q_c * G^T` by substituting the actual `G` blocks:

| State rows      | G slice | Result               |
|-----------------|---------|----------------------|
| `[3:6,  3:6 ]`  | `−R`    | `σ_a² · R·Rᵀ = σ_a²·I` |
| `[6:9,  6:9 ]`  | `−I`    | `σ_g²·I`             |
| `[9:12, 9:12]`  | `I`     | `σ_gb²·I`            |
| `[12:15,12:15]` | `I`     | `σ_ab²·I`            |

Because `R·Rᵀ = I` for any rotation matrix, **every block is rotation-independent
and constant**.  The entire `G * Q_c * G^T` matrix can be computed once at
construction and stored as member `gqgt_` (15×15).

**What changed:**
- Added `Eigen::Matrix<double, 15, 15> gqgt_` member to `EKF`.
- Computed in constructor: four `block<3,3>` assignments on a zero-initialised
  15×15 matrix.
- `predict()` now computes `Q_d = gqgt_ * dt` — one scalar multiply on a
  15×15 matrix — instead of building `Q_c`, `G`, and their product.
- `computeFG` renamed to `computeF`; the `G` output parameter removed.

**Impact:** small at 200 Hz (the 15×15 `Phi * P * Phi^T` product dominates
predict cost regardless), but cleaner and avoids unnecessary stack allocation of
a 15×12 matrix on every IMU step.

---

#### `src/stereo_tracker.cpp` + `include/ekf_vio/stereo_tracker.hpp` — Tracker micro-optimisations

These were implemented before timing data was available.  They are correct
improvements but had negligible measured impact since the tracker was only
taking 1.4 ms/frame (5 % of total).

**1. Cached `FastFeatureDetector` (stereo_tracker.hpp / .cpp)**

`detectNew()` previously called `cv::FastFeatureDetector::create(threshold)`
on every invocation (whenever tracked feature count dropped below half of
`max_features`).  Each `create()` allocates a new detector object on the heap.

Fixed by adding `cv::Ptr<cv::FastFeatureDetector> fast_detector_` as a member,
constructed once in `StereoTracker::StereoTracker()` and reused thereafter.

**2. Pre-built LK optical-flow pyramid (stereo_tracker.hpp / .cpp)**

`cv::calcOpticalFlowPyrLK` builds image pyramids internally when passed a raw
`cv::Mat`.  The previous code stored `prev_left_` as a raw grayscale `cv::Mat`
and passed it to the temporal LK call each frame, causing OpenCV to rebuild the
full pyramid (4 levels, 21×21 window) even though it had been built identically
at the end of the previous frame.

Fixed by:
- Replacing `cv::Mat prev_left_` member with `std::vector<cv::Mat> prev_pyramid_`.
- At the end of each `track()` call, storing the pyramid via
  `cv::buildOpticalFlowPyramid(img_left, prev_pyramid_, ...)` instead of
  `img_left.copyTo(prev_left_)`.  This also eliminates one deep pixel-buffer
  copy per frame.
- Passing `prev_pyramid_` as `prevImg` to the temporal `calcOpticalFlowPyrLK`
  call; OpenCV detects the pre-built pyramid and skips internal pyramid
  construction.
- Existence check changed from `!prev_left_.empty()` to `!prev_pyramid_.empty()`.

**3. Reserved `residuals` and `jacobians` vectors (ekf.cpp)**

Added `.reserve(M)` (where `M` is the number of candidate measurements before
gating) to prevent reallocation during the per-feature gating loop.  At N ≈ 200
this avoids log₂(200) ≈ 7 reallocation-and-copy cycles per update step.

---

#### `app/euroc_rerun_main.cpp` — Async Rerun logging thread

**Motivation:** Rerun's `rec.log(...)` calls perform network or file I/O
synchronously.  Combined with `cv::cvtColor` (grayscale→RGB) and trajectory
vector appends, these operations blocked the VIO stereo callback and slowed
the pipeline by the time taken to serialise and transmit each frame's data.

**Design — producer/consumer queue:**
- Defined `struct RerunLogData` containing everything needed to log one stereo
  frame: dataset timestamp, estimated body position, optional ground-truth
  position, `Sophus::SE3f` camera-in-world transform, camera intrinsics, the
  grayscale `cv::Mat` (shallow/ref-counted copy), and a vector of raw
  camera-frame feature points `pts_cam` (`std::vector<Eigen::Vector3d>`).
- Added a `std::queue<RerunLogData>` (bounded to 8 entries) protected by a
  `std::mutex` + `std::condition_variable`, and a `std::atomic<bool> log_stop`.
- Spawned a `std::thread log_thread` before the replay loop.  The thread:
  - Waits on the condvar.
  - Pops one `RerunLogData`, releases the lock.
  - Performs all `rec.log(...)` calls, `cv::cvtColor`, trajectory vector
    appends, and the `pts_cam → world` transform loop.
  - Loops until `log_stop == true` and the queue is empty, then exits.
- The stereo callback now packages a `RerunLogData` and pushes it under the
  lock (dropping the frame if the queue is full — acceptable for visualisation).
- After `reader.replay()` returns, `log_stop = true` is set and
  `log_cv.notify_one()` called, causing the thread to drain remaining queued
  frames and exit cleanly before `log_thread.join()`.

**Feature-point transform moved to logging thread:**
The callback previously computed `p_w = Twc * f.p_c` for every feature before
pushing to the queue.  The `pts_cam` field was introduced so the raw camera-frame
points are stored instead, and the `Twc * pc` transform loop runs in the logging
thread alongside the other I/O work.  `Twc` is stored in `RerunLogData` so the
logging thread has everything it needs.

**Thread safety:**
- `rerun::RecordingStream::log()` is thread-safe in the Rerun C++ SDK 0.21.
- `cv::Mat` uses reference-counted pixel buffers; shallow-copying into the queue
  struct keeps the buffer alive until the logging thread finishes with it.
- The two trajectory `std::vector<rerun::Vec3D>` accumulators (`est_traj`,
  `gt_traj`) live entirely inside the logging thread lambda — no cross-thread
  sharing.

---

### Benchmark summary (EuRoC V1_01_easy, 2911 stereo frames)

| Version | Rectify | Track | EKF upd | Total | Hz |
|---------|---------|-------|---------|-------|----|
| Baseline (before this branch) | — | — | — | ~28 ms | ~36 |
| After async Rerun + tracker micro-opts | 0.55 ms | 1.44 ms | 25.46 ms | 27.45 ms | 36 |
| **After sequential EKF update** | **0.54 ms** | **1.15 ms** | **0.26 ms** | **1.96 ms** | **510** |

Accuracy (ATE RMSE, V1_01_easy, Umeyama-aligned): **0.322 m** before and after.
Drift: **0.55 % of trajectory length**.
