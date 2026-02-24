/**
 * /workspace/src/ekf-vio/src/ekf.cpp
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

#include "ekf_vio/ekf.hpp"

#include <cmath>
#include <ekf_vio/math_utils.hpp>

namespace ekf_vio {

using namespace math;

// ---------------------------------------------------------------------------
EKF::EKF(const StereoCamera& cam, const NoiseParams& noise)
    : cam_(cam), noise_(noise) {}

// ---------------------------------------------------------------------------
// PREDICT step
// ---------------------------------------------------------------------------
void EKF::predict(const ImuData& imu, double dt) {
  // 1. Bias-corrected measurements
  const Eigen::Vector3d omega_c = imu.gyro - state_.b_g;
  const Eigen::Vector3d a_c = imu.accel - state_.b_a;

  // 2. Integrate state (RK4)
  PVQ pvq{state_.p, state_.v, state_.q};
  pvq = integrateRK4(pvq, omega_c, a_c, dt);
  state_.p = pvq.p;
  state_.v = pvq.v;
  state_.q = pvq.q;

  // 3. Compute error-state Jacobian F (15×15) and noise matrix G (15×12)
  Eigen::Matrix<double, 15, 15> F;
  Eigen::Matrix<double, 15, 12> G;
  computeFG(omega_c, a_c, F, G);

  // 4. Discretise  Phi ≈ I + F*dt,  Q_d = G * Q_c * G^T * dt
  const Eigen::Matrix<double, 15, 15> Phi =
      Eigen::Matrix<double, 15, 15>::Identity() + F * dt;

  // Continuous noise spectral density diagonal
  Eigen::Matrix<double, 12, 12> Q_c = Eigen::Matrix<double, 12, 12>::Zero();
  Q_c.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_gyro * noise_.sigma_gyro);
  Q_c.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_accel * noise_.sigma_accel);
  Q_c.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() *
                          (noise_.sigma_gyro_bias * noise_.sigma_gyro_bias);
  Q_c.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() *
                          (noise_.sigma_accel_bias * noise_.sigma_accel_bias);

  const Eigen::Matrix<double, 15, 15> Q_d = G * Q_c * G.transpose() * dt;

  // 5. Propagate covariance  P ← Phi P Phi^T + Q_d
  state_.P = Phi * state_.P * Phi.transpose() + Q_d;

  // Symmetrise (numerical insurance)
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// UPDATE step — stereo reprojection
// ---------------------------------------------------------------------------
void EKF::update(const std::vector<Feature>& features) {
  if (features.empty()) return;

  // Camera-to-world rotation and position (for point projection)
  const Eigen::Matrix3d R_cw =
      (state_.q.toRotationMatrix() * cam_.T_cam_imu.rotation()).transpose();
  const Eigen::Vector3d p_cam_w =
      state_.q.toRotationMatrix() * cam_.T_cam_imu.translation() + state_.p;

  // Stack residuals and Jacobians over all features
  const int N = static_cast<int>(features.size());
  Eigen::VectorXd z_all(4 * N);     // residuals
  Eigen::MatrixXd H_all(4 * N, 15); // Jacobians
  H_all.setZero();

  for (int i = 0; i < N; ++i) {
    const Feature& f = features[i];

    // 3-D point in world frame (triangulated in camera frame, transform)
    const Eigen::Vector3d p_w =
        state_.q.toRotationMatrix() * cam_.T_cam_imu.rotation() * f.p_c +
        p_cam_w;

    // Expected projections
    double eu_l, ev_l, eu_r, ev_r;
    project(f.p_c, eu_l, ev_l, eu_r, ev_r);

    // Residual  z - h(x)
    z_all(4 * i + 0) = f.u_l - eu_l;
    z_all(4 * i + 1) = f.v_l - ev_l;
    z_all(4 * i + 2) = f.u_r - eu_r;
    z_all(4 * i + 3) = f.v_r - ev_r;

    H_all.block<4, 15>(4 * i, 0) = measurementJacobian(f, R_cw, p_w);
  }

  // Measurement noise covariance  R = σ²_pix * I
  const double sig2 = noise_.sigma_pixel * noise_.sigma_pixel;
  const Eigen::MatrixXd R_mat = Eigen::MatrixXd::Identity(4 * N, 4 * N) * sig2;

  // Kalman gain  K = P H^T (H P H^T + R)^{-1}
  const Eigen::MatrixXd S = H_all * state_.P * H_all.transpose() + R_mat;
  const Eigen::MatrixXd K = state_.P * H_all.transpose() * S.inverse();

  // Error state update  δx = K * z
  const Eigen::VectorXd dx = K * z_all;

  // Apply error state to nominal state (boxplus for orientation)
  state_.p += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.q = boxplus(state_.q, dx.segment<3>(6));
  state_.b_g += dx.segment<3>(9);
  state_.b_a += dx.segment<3>(12);

  // Covariance update  (Joseph form for numerical stability)
  const Eigen::Matrix<double, 15, 15> IKH =
      Eigen::Matrix<double, 15, 15>::Identity() - K * H_all;
  state_.P = IKH * state_.P * IKH.transpose() + K * R_mat * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// RK4 IMU integration
// ---------------------------------------------------------------------------
EKF::PVQ EKF::integrateRK4(const PVQ& pvq, const Eigen::Vector3d& omega_c,
                           const Eigen::Vector3d& a_c, double dt) const {
  const Eigen::Vector3d g = gravity();
  // Derivative function: given state, return (ṗ, v̇, q̇ axis-angle)
  auto deriv = [&](const PVQ& s) -> PVQ {
    const Eigen::Matrix3d R = s.q.toRotationMatrix();
    return {s.v, R * a_c + g,
            Eigen::Quaterniond(0.0, 0.5 * omega_c.x(), 0.5 * omega_c.y(),
                               0.5 * omega_c.z())};
    // Note: q̇ = 0.5 * q ⊗ [0, ω].  We encode the ω part here; multiply below.
  };

  // For quaternion we use direct integration:  Δq = expSO3(ω * dt)
  // k1..k4 for p and v; quaternion handled separately via expSO3
  const Eigen::Matrix3d R = pvq.q.toRotationMatrix();

  // k1
  const Eigen::Vector3d dp1 = pvq.v;
  const Eigen::Vector3d dv1 = R * a_c + g;

  // k2 (use mid-point velocity from k1)
  const Eigen::Vector3d v2 = pvq.v + 0.5 * dt * dv1;
  const Eigen::Matrix3d R2 = expSO3(omega_c * 0.5 * dt) * R;
  const Eigen::Vector3d dp2 = v2;
  const Eigen::Vector3d dv2 = R2 * a_c + g;

  // k3 (same mid-point rotation as k2)
  const Eigen::Vector3d v3 = pvq.v + 0.5 * dt * dv2;
  const Eigen::Vector3d dp3 = v3;
  const Eigen::Vector3d dv3 = R2 * a_c + g;

  // k4 (full step rotation)
  const Eigen::Vector3d v4 = pvq.v + dt * dv3;
  const Eigen::Matrix3d R4 = expSO3(omega_c * dt) * R;
  const Eigen::Vector3d dp4 = v4;
  const Eigen::Vector3d dv4 = R4 * a_c + g;

  PVQ next;
  next.p = pvq.p + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
  next.v = pvq.v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
  next.q = Eigen::Quaterniond(expSO3(omega_c * dt) * R).normalized();
  return next;
}

// ---------------------------------------------------------------------------
// Error-state Jacobian F (continuous time, 15×15)
//
// Notation: R = R_{w←b} (body-to-world rotation matrix)
//
//  ṗ  =  v
//  v̇  =  R*(a_c) + g     →  ∂v̇/∂δθ = -R[a_c]×,  ∂v̇/∂δb_a = -R
//  θ̇  =  ω_c             →  ∂θ̇/∂δθ = -[ω_c]×,   ∂θ̇/∂δb_g = -I
//  ḃ_g = 0,  ḃ_a = 0
// ---------------------------------------------------------------------------
void EKF::computeFG(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                    Eigen::Matrix<double, 15, 15>& F,
                    Eigen::Matrix<double, 15, 12>& G) const {
  F.setZero();
  G.setZero();

  const Eigen::Matrix3d R = state_.q.toRotationMatrix();

  // ṗ = v
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  // v̇ = R*a_c + g  (wrt error state)
  F.block<3, 3>(3, 6) = -R * skew(a_c); // ∂/∂δθ
  F.block<3, 3>(3, 12) = -R;            // ∂/∂δb_a
  // θ̇ = ω_c
  F.block<3, 3>(6, 6) = -skew(omega_c);               // ∂/∂δθ
  F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity(); // ∂/∂δb_g

  // Noise input matrix G (maps 12-dim noise → 15-dim error state)
  //  n = [n_g, n_a, n_bg, n_ba]
  G.block<3, 3>(3, 3) = -R;                           // accel noise → velocity
  G.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity(); // gyro noise → angle
  G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // gyro bias noise
  G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // accel bias noise
}

// ---------------------------------------------------------------------------
// Project 3-D point in left camera frame to stereo pixel pair
// ---------------------------------------------------------------------------
void EKF::project(const Eigen::Vector3d& p_c, double& u_l, double& v_l,
                  double& u_r, double& v_r) const {
  const double inv_z = 1.0 / p_c.z();
  u_l = cam_.fx * p_c.x() * inv_z + cam_.cx;
  v_l = cam_.fy * p_c.y() * inv_z + cam_.cy;
  u_r = cam_.fx * (p_c.x() - cam_.baseline) * inv_z + cam_.cx;
  v_r = v_l; // epipolar aligned (rectified)
}

// ---------------------------------------------------------------------------
// Measurement Jacobian  H (4×15)
//
// Observation model (stereo):
//   z = [u_l, v_l, u_r, v_r]
//   p_c = R_{c←w} * (p_w - p_imu_w) - R_{c←b} * p_{c←b}
//       = R_{c←w} * p_w + t_c
//
// For tightly-coupled EKF we propagate H from the 3-D point
// to the error-state variables (p, v=0, θ, b_g=0, b_a=0).
// ---------------------------------------------------------------------------
Eigen::Matrix<double, 4, 15>
EKF::measurementJacobian(const Feature& f, const Eigen::Matrix3d& R_cw,
                         const Eigen::Vector3d& p_w) const {

  const Eigen::Vector3d pc = f.p_c; // 3-D in camera frame
  const double z = pc.z();
  const double z2 = z * z;

  // ∂[u_l, v_l] / ∂p_c  (2×3)
  Eigen::Matrix<double, 2, 3> J_proj_l;
  J_proj_l << cam_.fx / z, 0.0, -cam_.fx * pc.x() / z2, 0.0, cam_.fy / z,
      -cam_.fy * pc.y() / z2;

  // ∂[u_r, v_r] / ∂p_c  (2×3) — only u_r differs (x-baseline)
  Eigen::Matrix<double, 2, 3> J_proj_r;
  J_proj_r << cam_.fx / z, 0.0, -cam_.fx * (pc.x() - cam_.baseline) / z2, 0.0,
      cam_.fy / z, -cam_.fy * pc.y() / z2;

  // ∂p_c / ∂δp = -R_{c←w}    (3×3)
  // Increasing IMU position moves the feature in the opposite camera direction
  const Eigen::Matrix3d dp_c_dp = -R_cw;

  // ∂p_c / ∂δθ = [p_c + R_cb·t_bc]× · R_cw  (full form)
  // Approximation dropping lever arm (valid when camera-IMU offset is small):
  //   ≈ [p_c]× · R_cw
  const Eigen::Matrix3d dp_c_dtheta = skew(pc) * R_cw;

  // Stack into H (4×15)
  Eigen::Matrix<double, 4, 15> H;
  H.setZero();

  // Columns 0:2  → position δp
  H.block<2, 3>(0, 0) = J_proj_l * dp_c_dp;
  H.block<2, 3>(2, 0) = J_proj_r * dp_c_dp;

  // Columns 6:8  → orientation δθ
  H.block<2, 3>(0, 6) = J_proj_l * dp_c_dtheta;
  H.block<2, 3>(2, 6) = J_proj_r * dp_c_dtheta;

  return H;
}

} // namespace ekf_vio
