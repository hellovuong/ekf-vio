/**
 * /workspace/src/ekf-vio/include/ekf_vio/math_utils.hpp
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
#include <sophus/so3.hpp>

namespace ekf_vio {
namespace math {

// ---------------------------------------------------------------------------
// Skew-symmetric matrix [v]×  such that [v]× u = v × u
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  return Sophus::SO3d::hat(v);
}

// ---------------------------------------------------------------------------
// Rotation vector → rotation matrix  (exp map on SO3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d expSO3(const Eigen::Vector3d &omega) {
  return Sophus::SO3d::exp(omega).matrix();
}

// ---------------------------------------------------------------------------
// Rotation matrix → rotation vector  (log map on SO3)
// ---------------------------------------------------------------------------
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d &R) {
  return Sophus::SO3d(R).log();
}

// ---------------------------------------------------------------------------
// Quaternion ⊞ error-rotation-vector   (boxplus on SO3)
//   Body-frame (right-multiply) convention:  R_new = R * exp([δθ]×)
//   q_new = q ⊗ δq
// ---------------------------------------------------------------------------
inline Eigen::Quaterniond boxplus(const Eigen::Quaterniond &q,
                                  const Eigen::Vector3d &dtheta) {
  return (Sophus::SO3d(q) * Sophus::SO3d::exp(dtheta)).unit_quaternion();
}

// ---------------------------------------------------------------------------
// Left Jacobian of SO(3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d &omega) {
  return Sophus::SO3d::leftJacobian(omega);
}

// ---------------------------------------------------------------------------
// Inverse left Jacobian of SO(3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d invLeftJacobianSO3(const Eigen::Vector3d &omega) {
  return Sophus::SO3d::leftJacobianInverse(omega);
}

// ---------------------------------------------------------------------------
// Gravity vector in world frame (ENU: up = +Z)
// ---------------------------------------------------------------------------
inline Eigen::Vector3d gravity() { return {0.0, 0.0, -9.81}; }

} // namespace math
} // namespace ekf_vio
