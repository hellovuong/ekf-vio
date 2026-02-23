# EKF Visual-Inertial Odometry

A tightly-coupled **EKF-VIO** system for a **stereo camera + 6-DOF IMU** rig, written in C++17 with Eigen and ROS 2.

---

## Architecture

```
/imu/data  ──► [EKF PREDICT]  @ ~200 Hz
                     │
/cam/left  ──► [StereoTracker] ──► triangulated features
/cam/right         │
                   ▼
             [EKF UPDATE]     @ ~30 Hz
                   │
             /vio/odometry
```

---

## State Vector

The filter maintains a **16-dimensional nominal state** and a **15-dimensional error state** (used for the covariance matrix):

| Variable | Symbol | Dim | Description |
|---|---|---|---|
| Position | `p` | 3 | World frame (ENU) |
| Velocity | `v` | 3 | World frame |
| Orientation | `q` | 4 | Quaternion, world←body |
| Gyro bias | `b_g` | 3 | Body frame |
| Accel bias | `b_a` | 3 | Body frame |

The **error state** δx ∈ ℝ¹⁵ uses a rotation-vector for orientation error (SO(3) parameterisation), shrinking the quaternion's 4 DOF down to 3.

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

**Measurement Jacobian H (4×15):**

```
H = ∂z/∂δx = [J_proj · ∂p_c/∂p,   0,   J_proj · ∂p_c/∂δθ,   0,  0]
```

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

## File Layout

```
ekf_vio/
├── include/ekf_vio/
│   ├── types.hpp           # State, ImuData, Feature, StereoCamera structs
│   ├── math_utils.hpp      # skew(), expSO3(), logSO3(), boxplus(), gravity()
│   ├── ekf.hpp             # EKF class declaration
│   └── stereo_tracker.hpp  # StereoTracker class declaration
├── src/
│   ├── ekf.cpp             # Predict + Update implementations
│   ├── stereo_tracker.cpp  # LK tracking, triangulation, RANSAC
│   └── vio_node.cpp        # ROS 2 node
├── config/
│   └── euroc.yaml          # Parameters for EuRoC dataset
├── launch/
│   └── vio.launch.py
├── CMakeLists.txt
└── package.xml
```

---

## Building

```bash
# 1. Place in your ROS 2 workspace
cd ~/ros2_ws/src
cp -r /path/to/ekf_vio .

# 2. Install dependencies
sudo apt install ros-$ROS_DISTRO-cv-bridge       \
                 ros-$ROS_DISTRO-message-filters  \
                 libeigen3-dev libopencv-dev

# 3. Build
cd ~/ros2_ws
colcon build --packages-select ekf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

---

## Running with EuRoC

```bash
# Terminal 1 — launch VIO
ros2 launch ekf_vio vio.launch.py config:=src/ekf_vio/config/euroc.yaml

# Terminal 2 — play bag
ros2 bag play MH_01_easy.bag

# Terminal 3 — visualise
rviz2
# Add: TF, Odometry (/vio/odometry), Image (/camera/left/image_raw)
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

Use **Allan variance** plots from `imu_utils` or `kalibr` to get your actual IMU noise densities.

---

## Known Limitations & Next Steps

- **No loop closure** — drift accumulates over long runs.  Add a pose-graph backend (e.g. g2o / GTSAM).
- **Linearisation errors** — For aggressive motion consider an **MSCKF** (multi-state constraint) or **iSAM2**.
- **Initialisation** — Currently sets identity attitude.  Add a proper static initialiser that averages the first N IMU readings to estimate gravity direction.
- **Camera-IMU calibration** — Use `kalibr` to get accurate `T_cam_imu` and time-offset.
- **Rolling shutter** — If your camera has a rolling shutter, add per-row time correction.
