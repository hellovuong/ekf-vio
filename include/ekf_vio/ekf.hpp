/**
 * /workspace/src/ekf-vio/include/ekf_vio/ekf.hpp
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

#include "ekf_vio/types.hpp"
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
    double sigma_gyro = 1.6968e-4;       // rad/s/√Hz
    double sigma_accel = 2.0000e-3;      // m/s²/√Hz
    double sigma_gyro_bias = 1.9393e-5;  // rad/s²/√Hz
    double sigma_accel_bias = 3.0000e-5; // m/s³/√Hz
    double sigma_pixel = 1.5;            // pixels (reprojection noise)
  };

  explicit EKF(const StereoCamera& cam, const NoiseParams& noise);

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
    Eigen::Quaterniond q;
  };
  PVQ integrateRK4(const PVQ& pvq,
                   const Eigen::Vector3d& omega_c, // corrected gyro
                   const Eigen::Vector3d& a_c,     // corrected accel
                   double dt) const;

  // -----------------------------------------------------------------------
  // Continuous-time error-state Jacobian  F (15×15)
  // and noise-input matrix G (15×12)
  //
  // Discretised as:
  //   Phi  = I + F*dt  (first-order, sufficient at high IMU rates)
  //   Q_d  = G * Q_c * G^T * dt
  // -----------------------------------------------------------------------
  void computeFG(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                 Eigen::Matrix<double, 15, 15>& F,
                 Eigen::Matrix<double, 15, 12>& G) const;

  // -----------------------------------------------------------------------
  // Project a 3-D point in camera frame to (left, right) pixel pairs
  // -----------------------------------------------------------------------
  void project(const Eigen::Vector3d& p_c, double& u_l, double& v_l,
               double& u_r, double& v_r) const;

  // -----------------------------------------------------------------------
  // Compute measurement Jacobian  H (2×15) for one feature
  // (left u,v  or we can use all 4 observations: extend to 4×15)
  // -----------------------------------------------------------------------
  Eigen::Matrix<double, 4, 15>
  measurementJacobian(const Feature& f,
                      const Eigen::Matrix3d& R_cw, // rotation world→cam
                      const Eigen::Vector3d& p_w   // 3-D point in world frame
  ) const;

  State state_;
  StereoCamera cam_;
  NoiseParams noise_;
};

} // namespace ekf_vio
