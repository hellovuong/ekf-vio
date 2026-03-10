# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All builds use a ROS 2 Jazzy workspace via `colcon`:

```bash
# Standard build
colcon build --symlink-install

# Build with tests enabled
colcon build --cmake-args -DBUILD_TESTING=ON

# Build with Rerun visualization support
colcon build --cmake-args -DWITH_RERUN=ON

# Build with coverage
colcon build --cmake-args -DBUILD_TESTING=ON -DCMAKE_CXX_FLAGS="--coverage"
```

Docker multi-stage targets mirror the CI flow:
```bash
# Build and run tests (CI target)
docker build --target test -t ekf_vio:test .

# Build with Rerun visualizer
docker build --target viz -t ekf_vio:viz .

# Dev environment
docker build --target dev -t ekf_vio:dev .
```

## Testing

```bash
# Run all tests
colcon test && colcon test-result --verbose

# Run a single test binary directly (after build)
./build/ekf_vio/test/ekf_test
./build/ekf_vio/test/stereo_tracker_test
./build/ekf_vio/test/stereo_vo_test

# Run a single GTest case
./build/ekf_vio/test/ekf_test --gtest_filter="EKFTest.StationaryIMUKeepsPosition"
```

## Linting and Formatting

```bash
# Run clang-format on all C++ files (auto-fix)
clang-format -i include/ekf_vio/*.hpp src/*.cpp app/*.cpp test/*.cpp

# Run all pre-commit hooks (format + license headers + secrets)
pre-commit run --all-files

# clang-tidy runs automatically during colcon build (WarningsAsErrors)
```

## Running the System

```bash
# Standalone EuRoC runner (EKF-VIO)
./install/ekf_vio/lib/ekf_vio/euroc_runner <euroc_dataset_path> config/euroc.yaml

# VO-only runner
./install/ekf_vio/lib/ekf_vio/euroc_vo_runner <euroc_dataset_path> config/euroc.yaml

# Rerun visualization runner
./install/ekf_vio/lib/ekf_vio/euroc_rerun_runner <euroc_dataset_path> config/euroc.yaml --save output.rrd
./install/ekf_vio/lib/ekf_vio/euroc_rerun_runner <euroc_dataset_path> config/euroc.yaml --connect localhost:9876

# ROS 2 node
ros2 launch ekf_vio vio.launch.py
```

## Architecture

### Pipeline

IMU data (~200 Hz) feeds `EKF::predict()` using 4th-order Runge-Kutta integration. Stereo image pairs (~30 Hz) go through:
1. `StereoRectifier` — OpenCV rectification from YAML calibration
2. `StereoTracker` — FAST detection, Lucas-Kanade optical flow, F-matrix RANSAC outlier rejection, stereo triangulation
3. `StereoVO` — Keyframe-based PnP RANSAC + 3D-3D SVD alignment, emits `T_{world←cam}`
4. `EKF::update()` — Stereo reprojection measurement update (4 residuals/feature, Joseph form)
5. `EKF::updateFromPose()` — Optional loose-coupling of VO pose estimate

### State Vector

- **16D nominal state:** `T_wb` (Sophus::SE3d, world←body), velocity (3), gyro bias (3), accel bias (3)
- **15D error state:** position (3), velocity (3), rotation (3, SO(3) tangent), gyro bias (3), accel bias (3)
- Covariance `P` is 15×15; rotation error uses body-frame right-multiply convention via `boxplus()`

### Key Files

| File | Role |
|------|------|
| `include/ekf_vio/types.hpp` | `State`, `ImuData`, `Feature`, `StereoCamera`, `GroundTruth` |
| `include/ekf_vio/math_utils.hpp` | `skew()`, `expSO3()`, `logSO3()`, `boxplus()`, `leftJacobianSO3()` |
| `include/ekf_vio/ekf.hpp` + `src/ekf.cpp` | EKF predict (RK4) and update (reprojection) |
| `include/ekf_vio/stereo_tracker.hpp` + `src/stereo_tracker.cpp` | Feature tracking pipeline |
| `include/ekf_vio/stereo_vo.hpp` + `src/stereo_vo.cpp` | Keyframe-based visual odometry |
| `include/ekf_vio/config.hpp` | YAML config loader; sub-configs for Camera, IMU, Tracker, VO, EKF |
| `config/euroc.yaml` | Reference config for EuRoC MAV dataset |
| `app/euroc_main.cpp` | Main entry point for offline EuRoC evaluation |

### Coordinate Conventions

- `T_cam_imu` — transform from IMU frame to camera frame (camera←IMU)
- `T_wb` — world-to-body (world←body), i.e., body pose in world frame
- `T_wc = T_wb * T_cam_imu.inverse()` — camera pose in world frame
- Gravity is `{0, 0, -9.81}` (ENU, +Z up)

## Code Style

- Google C++ style, 2-space indent, 100-column limit (enforced by `.clang-format`)
- `const auto&` for references; `static_cast<float>(...)` not C-style casts
- All source files require SPDX header (enforced by pre-commit `insert-license`):
  ```cpp
  // Copyright (c) 2026, Long Vuong
  // SPDX-License-Identifier: BSD-3-Clause
  ```
- Namespace: `ekf_vio`; math utilities in `ekf_vio::math`
- clang-tidy runs with `WarningsAsErrors: '*'`; disabled per-target for third-party code (Rerun)

## Important Build Notes

- `src/live_coding/` contains untracked WIP files — the `live_coding` library target in CMakeLists.txt is intentional; leave it alone
- Rerun SDK (0.21.0) is fetched via CMake `FetchContent` when `WITH_RERUN=ON`
- Warning flags `-Wall -Wextra -Wpedantic -Werror` are applied `PRIVATE` per target (not transitively)
- Default build type is `RelWithDebInfo`
