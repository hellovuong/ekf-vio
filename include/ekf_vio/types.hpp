/**
 * /workspace/src/ekf-vio/include/ekf_vio/types.hpp
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace ekf_vio {

// ---------------------------------------------------------------------------
// State vector layout (16-dimensional):
//   [0:2]   position       p  (world frame)
//   [3:5]   velocity       v  (world frame)
//   [6:9]   quaternion     q  (world←body, stored as [qx,qy,qz,qw])
//   [10:12] gyro bias      b_g (body frame)
//   [13:15] accel bias     b_a (body frame)
//
// Error-state / covariance is 15-dimensional (quaternion error = 3D rotation
// vector via the "boxplus" convention, so we lose 1 DOF):
//   [0:2]   δp, [3:5] δv, [6:8] δθ, [9:11] δb_g, [12:14] δb_a
// ---------------------------------------------------------------------------

constexpr int STATE_DIM = 16; // full state
constexpr int ERR_DIM = 15;   // error-state (covariance lives here)

struct State {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();           // position
  Eigen::Vector3d v = Eigen::Vector3d::Zero();           // velocity
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // orientation
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();         // gyro bias
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();         // accel bias

  // Covariance in error-state space (15×15)
  Eigen::Matrix<double, ERR_DIM, ERR_DIM> P =
      Eigen::Matrix<double, ERR_DIM, ERR_DIM>::Identity() * 1e-6;
};

// IMU measurement bundle
struct ImuData {
  double timestamp;
  Eigen::Vector3d gyro;  // rad/s
  Eigen::Vector3d accel; // m/s²
};

// A triangulated stereo feature (left-camera frame)
struct Feature {
  int id;
  double u_l, v_l;     // left  pixel
  double u_r, v_r;     // right pixel
  Eigen::Vector3d p_c; // 3-D point in left-camera frame (after triangulation)
};

// Camera intrinsics & stereo extrinsics
struct StereoCamera {
  double fx, fy, cx, cy;       // left camera intrinsics
  double baseline;             // metres, right cam offset along -X_cam
  Eigen::Isometry3d T_cam_imu; // T_{cam←imu}
};

// Gravity constant (change if not at sea level or different planet ;)
constexpr double GRAVITY = 9.81;

} // namespace ekf_vio
