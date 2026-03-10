# EKF Visual-Inertial Odometry

[![CI](https://github.com/hellovuong/ekf-vio/actions/workflows/ci.yml/badge.svg)](https://github.com/hellovuong/ekf-vio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/hellovuong/ekf-vio/graph/badge.svg?token=D3YZ4MI6M4)](https://codecov.io/github/hellovuong/ekf-vio)

A tightly-coupled **EKF-VIO** system for a **stereo camera + 6-DOF IMU** rig,
written in modern C++20 with Eigen, Sophus, spdlog, and ROS 2.

Includes a standalone **keyframe-based stereo VO** pipeline and offline
evaluation tools for the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

---

## Architecture

```
/imu/data  ──► [EKF PREDICT]  @ ~200 Hz
                     │
/cam/left  ──► [StereoRectifier] ──► [StereoTracker] ──► triangulated features
/cam/right          │                       │
                    │              ┌────────┴──────────┐
                    ▼              ▼                    ▼
              [EKF UPDATE]   [StereoVO]          /vio/odometry
              (tightly-coupled  (keyframe PnP /
               reprojection)     3D-3D alignment)
```

---

## Benchmarks

Measured on **EuRoC V1_01_easy** (2911 stereo frames, 29 120 IMU samples)
running the standalone `euroc_runner` binary (`RelWithDebInfo`, single thread,
no visualisation).

### Runtime — per-stage breakdown

| Stage | ms / frame | Share |
|---|---|---|
| Stereo rectification | 0.54 ms | 28 % |
| Feature tracking (LK + RANSAC + triangulation) | 1.15 ms | 59 % |
| EKF visual update (sequential Kalman) | 0.26 ms | 13 % |
| **Total** | **1.96 ms** | — |
| **Throughput** | **510 Hz** | — |

### Accuracy — EuRoC V1_01_easy

```
GT  trajectory length :  58.56 m
Est trajectory length :  58.82 m
Scale factor          :  1.000

ATE — Absolute Trajectory Error
  RMSE   :  0.3222 m
  Mean   :  0.2947 m
  Median :  0.2670 m
  Max    :  0.6518 m

RPE — Relative Pose Error
  Translation RMSE  :  0.0415 m
  Translation Mean  :  0.0363 m
  Rotation    RMSE  :  0.1737 deg
  Rotation    Mean  :  0.1315 deg

Drift  :  0.55 % of trajectory length
```

---

## State Vector

The filter maintains a **16-dimensional nominal state** and a
**15-dimensional error state** (used for the covariance matrix):

| Variable | Symbol | Dim | Description |
|---|---|---|---|
| Position | `p` | 3 | World frame (ENU) |
| Velocity | `v` | 3 | World frame |
| Orientation | `q` | 4 | Quaternion, world←body |
| Gyro bias | `b_g` | 3 | Body frame |
| Accel bias | `b_a` | 3 | Body frame |

The **error state** δx ∈ ℝ¹⁵ uses a rotation-vector for orientation error
(SO(3) parameterisation via Sophus), shrinking the quaternion's 4 DOF down to 3.

---

## Mathematics

### Predict Step (IMU Mechanisation)

The IMU drives the filter at high rate.  Bias-corrected measurements are:

```
ω_c = ω_meas - b_g        (angular velocity)
a_c = a_meas - b_a        (specific force)
```

State derivatives (continuous time):

```
ṗ = v
v̇ = R·a_c + g_w
q̇ = q ⊗ [0, ω_c/2]      (quaternion kinematics)
ḃ_g = 0,  ḃ_a = 0        (random walk model)
```

Integration: **4th-order Runge-Kutta** for p, v, q.

#### Error-State Linearisation  F (15×15)

```
F = [ 0   I   0          0        0   ]   ← ṗ
    [ 0   0  -R[a_c]×    0       -R   ]   ← v̇
    [ 0   0  -[ω_c]×    -I        0   ]   ← θ̇
    [ 0   0   0          0        0   ]   ← ḃ_g
    [ 0   0   0          0        0   ]   ← ḃ_a
```

Covariance propagation:

```
Φ  = I + F·Δt            (first-order, valid at 200 Hz)
P  ← Φ P Φᵀ + G Q_c Gᵀ Δt
```

where G maps continuous noise (gyro, accel, bias) into the error state.

---

### Update Step (Stereo Reprojection)

For each tracked feature with 3-D position `p_c` in the left-camera frame:

**Left camera projection:**
```
u_l = fx · X/Z + cx
v_l = fy · Y/Z + cy
```

**Right camera projection** (rectified stereo, baseline b):
```
u_r = fx · (X - b)/Z + cx
v_r = v_l                       (epipolar aligned)
```

**Residual:** `z = [u_l, v_l, u_r, v_r]ᵀ - h(x)`

**Kalman Update (Joseph form for numerical stability):**
```
S = H P Hᵀ + R
K = P Hᵀ S⁻¹
δx  = K · z
P  = (I - KH) P (I - KH)ᵀ + K R Kᵀ
```

State boxplus (orientation via SO(3) exponential map):
```
q  ← expSO3(δθ) ⊗ q
p  ← p + δp
v  ← v + δv
```

---

## Feature Tracker

The `StereoTracker` uses:
1. **FAST** corner detection for new features (masked around existing tracks)
2. **Lucas-Kanade** optical flow for temporal tracking (left_prev → left_curr)
3. **LK flow** for stereo matching (left_curr → right_curr, constrained to same row)
4. **Fundamental matrix RANSAC** for outlier rejection
5. **Stereo triangulation** (rectified formula) for 3-D initialisation

---

## Stereo Visual Odometry

The `StereoVO` module provides a standalone keyframe-based pipeline:
1. Match tracked features against keyframe landmarks by persistent ID
2. Estimate motion via **3D-3D SVD alignment** (Horn's method) with RANSAC
3. Motion sanity check (reject implausible translation/rotation jumps)
4. Keyframe management (spawn new keyframe when tracking ratio drops)

The VO pose can be fused into the EKF via `updateFromPose()` for
loosely-coupled integration.

---

## File Layout

```
ekf-vio/
├── include/ekf_vio/
│   ├── types.hpp              # State, ImuData, Feature, StereoCamera structs
│   ├── math_utils.hpp         # skew(), expSO3(), logSO3(), boxplus() via Sophus
│   ├── ekf.hpp                # EKF class (predict + update)
│   ├── stereo_tracker.hpp     # LK tracking, triangulation, RANSAC
│   ├── stereo_vo.hpp          # Keyframe-based stereo VO
│   ├── stereo_rectifier.hpp   # Stereo rectification from calibration config
│   ├── euroc_reader.hpp       # EuRoC MAV dataset reader
│   ├── config.hpp             # YAML config loader (camera, IMU, tracker, VO, EKF)
│   └── logging.hpp            # Thin spdlog wrapper
├── src/
│   ├── ekf.cpp                # EKF predict + update implementations
│   ├── stereo_tracker.cpp     # LK tracking, triangulation, RANSAC
│   ├── stereo_vo.cpp          # Keyframe VO (PnP / 3D-3D alignment)
│   ├── stereo_rectifier.cpp   # OpenCV stereo rectification
│   └── euroc_reader.cpp       # CSV parsing + timeline merge
├── app/
│   ├── vio_node.cpp           # ROS 2 VIO node
│   ├── euroc_main.cpp         # Standalone EuRoC runner (EKF-VIO)
│   └── euroc_vo_main.cpp      # Standalone EuRoC runner (VO-only)
├── test/
│   ├── ekf_test.cpp           # EKF unit tests (GTest)
│   ├── stereo_tracker_test.cpp
│   └── stereo_vo_test.cpp
├── config/
│   └── euroc.yaml             # Parameters for EuRoC dataset
├── launch/
│   └── vio.launch.py          # ROS 2 launch file
├── scripts/
│   ├── evaluate_euroc.py      # ATE/RPE evaluation + plotting
│   └── compare_vo_vio.py      # Compare VO vs VIO trajectories
├── learning/
│   └── ba.cpp                 # Educational: two-view RGBD bundle adjustment
├── Dockerfile                 # Multi-stage: deps → build → test → dev
├── CMakeLists.txt
├── package.xml
└── LICENSE                    # BSD 3-Clause
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| Eigen3 | ≥ 3.3 | Linear algebra, quaternions |
| OpenCV | ≥ 4.0 | Feature detection, optical flow, stereo rectification |
| Sophus | — | SO(3)/SE(3) Lie group operations |
| spdlog | — | Structured logging (info, debug, warn) |
| yaml-cpp | — | Configuration file parsing |
| ROS 2 Jazzy | — | `rclcpp`, `sensor_msgs`, `cv_bridge`, `tf2_ros`, etc. |

---

## Building

### ROS 2 Workspace (full build)

```bash
# 1. Place in your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/hellovuong/ekf-vio.git

# 2. Install dependencies
sudo apt install \
  ros-jazzy-cv-bridge ros-jazzy-message-filters ros-jazzy-tf2-ros \
  libeigen3-dev libopencv-dev libspdlog-dev libyaml-cpp-dev

# 3. Install Sophus (header-only, not in Ubuntu apt repos)
git clone --depth 1 --branch 1.22.10 https://github.com/strasdat/Sophus.git /tmp/sophus
cmake -S /tmp/sophus -B /tmp/sophus/build -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
sudo cmake --build /tmp/sophus/build --target install

# 4. Build
cd ~/ros2_ws
colcon build --packages-select ekf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Docker

The Dockerfile is multi-stage (`deps` → `build` → `test` → `dev`):

```bash
# Local development (default target — includes nvim, rosdep, etc.)
docker build -t ekf-vio-dev .
docker run -it --rm -v $(pwd):/workspace/src/ekf-vio ekf-vio-dev

# Build + test only (same as CI)
docker build --target test -t ekf-vio-test .
```

---

## Running

### ROS 2 Node

```bash
# Terminal 1 — launch VIO
ros2 launch ekf_vio vio.launch.py config:=src/ekf_vio/config/euroc.yaml

# Terminal 2 — play bag
ros2 bag play MH_01_easy.bag

# Terminal 3 — visualise
rviz2
# Add: TF, Odometry (/vio/odometry), Image (/camera/left/image_raw)
```

### Standalone EuRoC Runner

```bash
# EKF-VIO (IMU + stereo)
./install/lib/ekf_vio/euroc_runner /path/to/MH_01_easy config/euroc.yaml

# VO-only (stereo, no IMU)
./install/lib/ekf_vio/euroc_vo_runner /path/to/MH_01_easy config/euroc.yaml
```

Both runners output `euroc_traj.csv` for evaluation.

### Evaluation

```bash
# Single trajectory evaluation with plots
python3 scripts/evaluate_euroc.py \
    --est euroc_traj.csv \
    --gt /path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
    --out results/MH_01

# Compare VO vs VIO
python3 scripts/compare_vo_vio.py vo_traj.csv vio_traj.csv gt_data.csv
```

---

## Testing

Tests use Google Test via `ament_cmake_gtest`:

```bash
cd ~/ros2_ws
colcon build --packages-select ekf_vio --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select ekf_vio --return-code-on-test-failure
colcon test-result --verbose
```

---

## Tuning Guide

| Parameter | Effect | Start value |
|---|---|---|
| `sigma_gyro` | Trust IMU gyro | 1.7e-4 rad/s/√Hz |
| `sigma_accel` | Trust IMU accel | 2e-3 m/s²/√Hz |
| `sigma_pixel` | Trust visual measurements | 1.5 px |
| `max_features` | More = more robust, slower | 200–400 |
| Initial `P` | Filter convergence speed | Identity × 1e-6 |

Use **Allan variance** plots from `imu_utils` or `kalibr` to get your actual
IMU noise densities.  All parameters are in `config/euroc.yaml`.

---

## Development

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit` and include:
- **clang-format** — C/C++ formatting
- **trailing-whitespace / end-of-file** — whitespace hygiene
- **gitleaks** — secret scanning
- **insert-license** — BSD-3-Clause SPDX header enforcement

### Formatting

```bash
# Check all C++ files
find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run -Werror

# Format in-place
find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

### Static Analysis

```bash
# Run clang-tidy (requires compile_commands.json)
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build/ekf_vio src/*.cpp
```

### CI

GitHub Actions runs on every push/PR to `main`:
- **build-and-test** — `docker build --target test` (the Dockerfile is the single source of truth for deps, build, and test)
- **lint** — all pre-commit hooks

Docker layer caching (`type=gha`) keeps subsequent CI runs fast.

---

## Known Limitations & Next Steps

- **No loop closure** — drift accumulates over long runs.  Add a pose-graph backend (e.g. g2o / GTSAM).
- **Linearisation errors** — For aggressive motion consider an **MSCKF** (multi-state constraint) or **iSAM2**.
- **Initialisation** — Currently sets identity attitude or uses ground truth.  Add a proper static initialiser that averages the first N IMU readings to estimate gravity direction.
- **Camera-IMU calibration** — Use `kalibr` to get accurate `T_cam_imu` and time-offset.
- **Rolling shutter** — If your camera has a rolling shutter, add per-row time correction.

---

## License

BSD 3-Clause — see [LICENSE](LICENSE).
