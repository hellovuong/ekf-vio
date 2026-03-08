// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace ekf_vio::math {

// ---------------------------------------------------------------------------
// Skew-symmetric matrix [v]×  such that [v]× u = v × u
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  return Sophus::SO3d::hat(v);
}

// ---------------------------------------------------------------------------
// Rotation vector → rotation matrix  (exp map on SO3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& omega) {
  return Sophus::SO3d::exp(omega).matrix();
}

// ---------------------------------------------------------------------------
// Rotation matrix → rotation vector  (log map on SO3)
// ---------------------------------------------------------------------------
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d& R) {
  return Sophus::SO3d(R).log();
}

// ---------------------------------------------------------------------------
// Quaternion ⊞ error-rotation-vector   (boxplus on SO3)
//   Body-frame (right-multiply) convention:  R_new = R * exp([δθ]×)
//   q_new = q ⊗ δq
// ---------------------------------------------------------------------------
inline Eigen::Quaterniond boxplus(const Eigen::Quaterniond& q, const Eigen::Vector3d& dtheta) {
  return (Sophus::SO3d(q) * Sophus::SO3d::exp(dtheta)).unit_quaternion();
}

// ---------------------------------------------------------------------------
// Left Jacobian of SO(3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& omega) {
  return Sophus::SO3d::leftJacobian(omega);
}

// ---------------------------------------------------------------------------
// Inverse left Jacobian of SO(3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d invLeftJacobianSO3(const Eigen::Vector3d& omega) {
  return Sophus::SO3d::leftJacobianInverse(omega);
}

// ---------------------------------------------------------------------------
// Gravity vector in world frame (ENU: up = +Z)
// ---------------------------------------------------------------------------
inline Eigen::Vector3d gravity() {
  return {0.0, 0.0, -9.81};
}

}  // namespace ekf_vio::math
