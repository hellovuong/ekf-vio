// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ekf_vio/types.hpp"

#include <unordered_map>
#include <vector>

namespace ekf_vio {

// ===========================================================================
//  EKF — Extended Kalman Filter for Visual-Inertial Odometry
//
//  Predict: IMU mechanisation (continuous-time, RK4 integration)
//           Covariance propagated via analytical Jacobian F
//
//  Update:  Stereo feature re-projection
//           Measurement Jacobian H computed analytically
// ===========================================================================
class EKF {
 public:
  // Process noise parameters (tune these!)
  struct NoiseParams {
    double sigma_gyro = 1.6968e-4;        // rad/s/√Hz
    double sigma_accel = 2.0000e-3;       // m/s²/√Hz
    double sigma_gyro_bias = 1.9393e-5;   // rad/s²/√Hz
    double sigma_accel_bias = 3.0000e-5;  // m/s³/√Hz
    double sigma_pixel = 1.5;             // pixels (reprojection noise)
    int landmark_max_age = 5;             // keep landmarks for N frames after last observation
  };

  explicit EKF(StereoCamera cam, const NoiseParams& noise);

  // -----------------------------------------------------------------------
  // Propagate state using IMU measurement over interval dt [seconds]
  // Call this for every IMU sample (~200 Hz)
  // -----------------------------------------------------------------------
  void predict(const ImuData& imu, double dt);

  // -----------------------------------------------------------------------
  // Update state using a set of triangulated stereo features
  // Call this for every camera frame (~30 Hz)
  // -----------------------------------------------------------------------
  void update(const std::vector<Feature>& features);

  // -----------------------------------------------------------------------
  // Update state using an external 6-DOF pose measurement (e.g. from VO).
  //   p_meas : measured position in world frame
  //   q_meas : measured orientation (world←body) as quaternion
  //   sigma_p: position measurement noise std-dev (metres)
  //   sigma_q: orientation measurement noise std-dev (radians)
  // -----------------------------------------------------------------------
  void updateFromPose(const Sophus::SE3d& T_meas, double sigma_p, double sigma_q);

  // Accessors
  const State& state() const { return state_; }
  State& state() { return state_; }

 private:
  // -----------------------------------------------------------------------
  // IMU Integration (RK4 on position, velocity, quaternion)
  // Returns updated (p, v, q) — biases are held constant in predict step
  // -----------------------------------------------------------------------
  struct PVQ {
    Eigen::Vector3d p, v;
    Sophus::SO3d R;
  };
  PVQ integrateRK4(const PVQ& pvq,
                   const Eigen::Vector3d& omega_c,  // corrected gyro
                   const Eigen::Vector3d& a_c,      // corrected accel
                   double dt) const;

  // -----------------------------------------------------------------------
  // Continuous-time error-state Jacobian  F (15×15)
  // and noise-input matrix G (15×12)
  //
  // Discretised as:
  //   Phi  = I + F*dt  (first-order, sufficient at high IMU rates)
  //   Q_d  = G * Q_c * G^T * dt
  // -----------------------------------------------------------------------
  void computeF(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                Eigen::Matrix<double, 15, 15>& F) const;

  // -----------------------------------------------------------------------
  // Project a 3-D point in camera frame to (left, right) pixel pairs
  // -----------------------------------------------------------------------
  void project(const Eigen::Vector3d& p_c, double& u_l, double& v_l, double& u_r,
               double& v_r) const;

  // Transform a 3-D point from camera frame to world frame using current state
  Eigen::Vector3d camToWorld(const Eigen::Vector3d& p_c) const;

  // Transform a 3-D point from world frame to camera frame using current state
  Eigen::Vector3d worldToCam(const Eigen::Vector3d& p_w) const;

  State state_;
  StereoCamera cam_;
  NoiseParams noise_;

  // G * Q_c * G^T is constant (rotation cancels out: R*R^T = I for every block).
  // Precomputed once in the constructor so predict() skips building Q_c and G
  // and the full matrix chain at 200 Hz.
  Eigen::Matrix<double, 15, 15> gqgt_;

  // Landmark map: feature ID → world position + last-seen frame
  struct Landmark {
    Eigen::Vector3d p_w;
    int last_seen_frame = 0;
  };
  std::unordered_map<int, Landmark> landmarks_;
  int frame_count_ = 0;
};

}  // namespace ekf_vio
