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

namespace ekf_vio {
namespace math {

// ---------------------------------------------------------------------------
// Skew-symmetric matrix [v]×  such that [v]× u = v × u
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// ---------------------------------------------------------------------------
// Rodrigues / small-angle rotation vector → rotation matrix
//   R = I + sinθ/θ [ω]× + (1-cosθ)/θ² [ω]×²
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < 1e-9) {
    return Eigen::Matrix3d::Identity() + skew(omega);
  }
  const Eigen::Matrix3d K = skew(omega / theta);
  return Eigen::Matrix3d::Identity() + std::sin(theta) * K +
         (1.0 - std::cos(theta)) * K * K;
}

// ---------------------------------------------------------------------------
// Rotation matrix → rotation vector (log map)
// ---------------------------------------------------------------------------
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d& R) {
  const double cosAngle =
      std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
  const double theta = std::acos(cosAngle);
  if (std::abs(theta) < 1e-9) return Eigen::Vector3d::Zero();
  return theta / (2.0 * std::sin(theta)) *
         Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                         R(1, 0) - R(0, 1));
}

// ---------------------------------------------------------------------------
// Quaternion ⊞ error-rotation-vector   (boxplus on SO3)
//   q_new = expSO3(dtheta) ⊗ q
// ---------------------------------------------------------------------------
inline Eigen::Quaterniond boxplus(const Eigen::Quaterniond& q,
                                  const Eigen::Vector3d& dtheta) {
  const Eigen::Quaterniond dq(expSO3(dtheta));
  return (dq * q).normalized();
}

// ---------------------------------------------------------------------------
// Left Jacobian of SO(3) — used for covariance propagation
//   J_l = sinθ/θ I + (1-sinθ/θ)[ω]×/θ + (1-cosθ)/θ² [ω]×²
//   (approx I + 0.5[ω]× for small angles)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < 1e-9) {
    return Eigen::Matrix3d::Identity() + 0.5 * skew(omega);
  }
  const Eigen::Matrix3d K = skew(omega / theta);
  return Eigen::Matrix3d::Identity() + (1.0 - std::cos(theta)) / theta * K +
         (1.0 - std::sin(theta) / theta) * K * K;
}

// ---------------------------------------------------------------------------
// Inverse left Jacobian of SO(3)
// ---------------------------------------------------------------------------
inline Eigen::Matrix3d invLeftJacobianSO3(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < 1e-9) {
    return Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
  }
  const Eigen::Matrix3d K = skew(omega / theta);
  return Eigen::Matrix3d::Identity() - 0.5 * theta * K +
         (1.0 - theta * (1.0 + std::cos(theta)) / (2.0 * std::sin(theta))) * K *
             K;
}

// ---------------------------------------------------------------------------
// Gravity vector in world frame (NED: down = +Z, or ENU: up = +Z)
// Change sign convention to match your world frame!
// Here we use ENU (up = +Z), so gravity = [0,0,-9.81]
// ---------------------------------------------------------------------------
inline Eigen::Vector3d gravity() {
  return Eigen::Vector3d(0.0, 0.0, -9.81);
}

} // namespace math
} // namespace ekf_vio
