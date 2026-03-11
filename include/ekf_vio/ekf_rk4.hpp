// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ekf_vio/types.hpp"

#include <unordered_map>
#include <vector>

namespace ekf_vio {

// ===========================================================================
//  EKFRk4 — EKF with full RK4 covariance/transition propagation
//
//  Identical update step to EKF.  Different predict step:
//
//    EKF (original):
//      - RK4 on state (p, v, R) only
//      - Φ ≈ I + F·dt           (first-order Euler)
//      - Q_d = GQG^T · dt       (zero-order hold)
//
//    EKFRk4 (this class):
//      - RK4 on (state, Φ, Q_d) simultaneously — one consistent integrator
//      - Φ̇ = F(x)·Φ,  Φ(0)=I   → O(dt⁵) error on Φ
//      - Q̇_d = F·P + P·F^T + GQG^T, P(0)=0  → O(dt⁵) error on Q_d
//      - IMU readings linearly interpolated for k2/k3 midpoint
//        (requires prev IMU buffered; falls back to ZOH on first step)
//
//  Reference: maplab imu_integrator (imu-integrator-inl.h)
//  https://github.com/ethz-asl/maplab/blob/master/algorithms/imu-integrator-rk4/include/imu-integrator/imu-integrator-inl.h
// ===========================================================================
class EKFRk4 {
 public:
  struct NoiseParams {
    double sigma_gyro = 1.6968e-4;
    double sigma_accel = 2.0000e-3;
    double sigma_gyro_bias = 1.9393e-5;
    double sigma_accel_bias = 3.0000e-5;
    double sigma_pixel = 1.5;
    int landmark_max_age = 5;
  };

  explicit EKFRk4(StereoCamera cam, const NoiseParams& noise);

  // Propagate state + covariance using one IMU measurement over dt [s].
  // Internally buffers the previous reading to build the bracketing pair
  // [imu_{t}, imu_{t+dt}] needed by the midpoint interpolation.
  void predict(const ImuData& imu, double dt);

  // Update using triangulated stereo features (identical to EKF::update).
  void update(const std::vector<Feature>& features);

  // Update from an external 6-DOF pose (loosely-coupled VO).
  void updateFromPose(const Sophus::SE3d& T_meas, double sigma_p, double sigma_q);

  const State& state() const { return state_; }
  State& state() { return state_; }

 private:
  // -----------------------------------------------------------------------
  // Position-velocity-rotation bundle for RK4 intermediate states
  // -----------------------------------------------------------------------
  struct PVQ {
    Eigen::Vector3d p, v;
    Sophus::SO3d R;
  };

  // -----------------------------------------------------------------------
  // Derivatives for one RK4 stage: state + matrix ODEs
  //   dp  = v
  //   dv  = R·a + g
  //   dΦ  = F·Φ
  //   dP  = F·P + P·F^T + GQG^T
  // -----------------------------------------------------------------------
  struct Deriv {
    Eigen::Vector3d dp;
    Eigen::Vector3d dv;
    Eigen::Matrix<double, 15, 15> dPhi;
    Eigen::Matrix<double, 15, 15> dP;
  };

  // Evaluate all four derivatives at a single (state, Φ, P, imu) point.
  Deriv evalDeriv(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c, const PVQ& pvq,
                  const Eigen::Matrix<double, 15, 15>& Phi,
                  const Eigen::Matrix<double, 15, 15>& P) const;

  // Continuous-time error-state Jacobian F(15×15) evaluated at rotation R.
  // Takes R explicitly so it can be called at RK4 intermediate states.
  void computeF(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                const Eigen::Matrix3d& R, Eigen::Matrix<double, 15, 15>& F) const;

  void project(const Eigen::Vector3d& p_c, double& u_l, double& v_l, double& u_r,
               double& v_r) const;
  Eigen::Vector3d camToWorld(const Eigen::Vector3d& p_c) const;
  Eigen::Vector3d worldToCam(const Eigen::Vector3d& p_w) const;

  State state_;
  StereoCamera cam_;
  NoiseParams noise_;

  // GQG^T is constant (R cancels: R·σ²·R^T = σ²·I). Precomputed once.
  Eigen::Matrix<double, 15, 15> gqgt_;

  // Buffered previous IMU reading for start-of-interval bracketing.
  ImuData prev_imu_{};
  bool has_prev_imu_ = false;

  struct Landmark {
    Eigen::Vector3d p_w;
    int last_seen_frame = 0;
  };
  std::unordered_map<int, Landmark> landmarks_;
  int frame_count_ = 0;
};

}  // namespace ekf_vio
