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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ekf_vio/logging.hpp>
#include <ekf_vio/math_utils.hpp>

namespace ekf_vio {

using namespace math;

// ---------------------------------------------------------------------------
EKF::EKF(const StereoCamera &cam, const NoiseParams &noise)
    : cam_(cam), noise_(noise) {}

// ---------------------------------------------------------------------------
// PREDICT step
// ---------------------------------------------------------------------------
void EKF::predict(const ImuData &imu, double dt) {
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
// UPDATE step — stereo reprojection against landmark map
// ---------------------------------------------------------------------------
void EKF::update(const std::vector<Feature> &features) {
  if (features.empty())
    return;

  ++frame_count_;

  // Precompute transforms
  const Eigen::Matrix3d R_wb = state_.q.toRotationMatrix(); // body→world
  const Eigen::Matrix3d R_ci = cam_.T_cam_imu.rotation();   // imu→cam
  const Eigen::Vector3d t_ci = cam_.T_cam_imu.translation();
  // World→camera: R_cw = R_ci * R_wb^T
  const Eigen::Matrix3d R_cw = R_ci * R_wb.transpose();

  // ------------------------------------------------------------------
  // Separate features into new (initialise landmark) vs tracked (measure)
  // ------------------------------------------------------------------
  std::vector<int> meas_indices;  // indices into features[] for measurement
  for (int i = 0; i < static_cast<int>(features.size()); ++i) {
    const Feature &f = features[i];
    auto it = landmarks_.find(f.id);
    if (it == landmarks_.end()) {
      // New landmark: triangulate and store in world frame
      if (f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
        landmarks_[f.id] = {camToWorld(f.p_c), frame_count_};
      }
    } else {
      it->second.last_seen_frame = frame_count_;
      meas_indices.push_back(i);
    }
  }

  const int M = static_cast<int>(meas_indices.size());
  if (M == 0) return;

  // ------------------------------------------------------------------
  // Build stacked residuals z and Jacobians H, with outlier rejection
  // ------------------------------------------------------------------
  const double sig2 = noise_.sigma_pixel * noise_.sigma_pixel;
  // Chi-squared threshold for 4-DOF at 95% confidence
  const double chi2_thresh = 9.488;

  std::vector<Eigen::Vector4d> residuals;
  std::vector<Eigen::Matrix<double, 4, 15>> jacobians;

  for (int k = 0; k < M; ++k) {
    const Feature &f = features[meas_indices[k]];
    const Eigen::Vector3d &p_w = landmarks_.at(f.id).p_w;

    // Predict camera-frame point from world landmark + current state
    const Eigen::Vector3d p_c_pred = worldToCam(p_w);

    // Skip landmarks behind the camera or too far
    if (p_c_pred.z() < 0.1 || p_c_pred.z() > 50.0)
      continue;

    // Predicted stereo projection
    double eu_l, ev_l, eu_r, ev_r;
    project(p_c_pred, eu_l, ev_l, eu_r, ev_r);

    // Residual  z - h(x)
    Eigen::Vector4d res;
    res(0) = f.u_l - eu_l;
    res(1) = f.v_l - ev_l;
    res(2) = f.u_r - eu_r;
    res(3) = f.v_r - ev_r;

    // --- Measurement Jacobian (4×15) ---
    const double z_c = p_c_pred.z();
    const double z_c2 = z_c * z_c;

    // ∂[u_l,v_l]/∂p_c
    Eigen::Matrix<double, 2, 3> J_l;
    J_l << cam_.fx / z_c, 0.0, -cam_.fx * p_c_pred.x() / z_c2,
           0.0, cam_.fy / z_c, -cam_.fy * p_c_pred.y() / z_c2;

    // ∂[u_r,v_r]/∂p_c  (right camera)
    Eigen::Matrix<double, 2, 3> J_r;
    // Horizontal stereo: u_r = fx*(X-b)/Z + cx,  v_r = fy*Y/Z + cy
    J_r << cam_.fx / z_c, 0.0, -cam_.fx * (p_c_pred.x() - cam_.baseline) / z_c2,
           0.0, cam_.fy / z_c, -cam_.fy * p_c_pred.y() / z_c2;

    // p_c = R_ci * R_wb^T * (p_w - p) + t_ci
    // ∂p_c/∂δp = -R_cw = -R_ci * R_wb^T
    const Eigen::Matrix3d dp_c_dp = -R_cw;

    // ∂p_c/∂δθ: perturb R_wb → R_wb * exp([δθ]×), then R_wb^T → exp(-[δθ]×)*R_wb^T
    //   δp_c = R_ci * [R_wb^T * (p_w - p)]× * δθ
    const Eigen::Vector3d p_imu = R_wb.transpose() * (p_w - state_.p);
    const Eigen::Matrix3d dp_c_dtheta = R_ci * skew(p_imu);

    // Stack into H (4×15)
    Eigen::Matrix<double, 4, 15> H_i;
    H_i.setZero();
    H_i.block<2, 3>(0, 0) = J_l * dp_c_dp;      // position
    H_i.block<2, 3>(2, 0) = J_r * dp_c_dp;
    H_i.block<2, 3>(0, 6) = J_l * dp_c_dtheta;   // orientation
    H_i.block<2, 3>(2, 6) = J_r * dp_c_dtheta;

    // Mahalanobis gating: reject outliers using innovation covariance
    const Eigen::Matrix4d R_i = Eigen::Matrix4d::Identity() * sig2;
    const Eigen::Matrix4d S_i = H_i * state_.P * H_i.transpose() + R_i;
    const double mahal = res.transpose() * S_i.inverse() * res;
    if (mahal > chi2_thresh)
      continue;

    residuals.push_back(res);
    jacobians.push_back(H_i);
  }

  // Cap to avoid oversized H matrices (keep top features by residual norm)
  const int max_meas = 200;
  if (static_cast<int>(residuals.size()) > max_meas) {
    // Keep features with smallest residuals (best matches)
    std::vector<int> idx(residuals.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + max_meas, idx.end(),
        [&](int a, int b) { return residuals[a].norm() < residuals[b].norm(); });
    std::vector<Eigen::Vector4d> r2;
    std::vector<Eigen::Matrix<double, 4, 15>> j2;
    r2.reserve(max_meas);
    j2.reserve(max_meas);
    for (int i = 0; i < max_meas; ++i) {
      r2.push_back(residuals[idx[i]]);
      j2.push_back(jacobians[idx[i]]);
    }
    residuals = std::move(r2);
    jacobians = std::move(j2);
  }

  const int N = static_cast<int>(residuals.size());
  if (N == 0) return;

  // Assemble stacked z and H
  Eigen::VectorXd z_all(4 * N);
  Eigen::MatrixXd H_all(4 * N, 15);
  H_all.setZero();
  for (int k = 0; k < N; ++k) {
    z_all.segment<4>(4 * k) = residuals[k];
    H_all.block<4, 15>(4 * k, 0) = jacobians[k];
  }

  // ------------------------------------------------------------------
  // Kalman update (LDLT for numerical stability)
  // ------------------------------------------------------------------
  const Eigen::MatrixXd R_mat = Eigen::MatrixXd::Identity(4 * N, 4 * N) * sig2;

  const Eigen::MatrixXd S = H_all * state_.P * H_all.transpose() + R_mat;
  // Solve K = P * H^T * S^{-1}  via  S^T * K^T = H * P^T  →  K^T = S^{-T} * H * P
  const Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);
  if (S_ldlt.info() != Eigen::Success) {
    get_logger()->warn("EKF visual update: LDLT decomposition failed");
    return;
  }
  const Eigen::MatrixXd K = state_.P * H_all.transpose() * S_ldlt.solve(Eigen::MatrixXd::Identity(4 * N, 4 * N));
  const Eigen::VectorXd dx = K * z_all;

  // NaN check on dx
  if (!dx.allFinite()) {
    get_logger()->warn("EKF visual update: dx contains NaN — skipping");
    return;
  }

  state_.p += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.q = boxplus(state_.q, dx.segment<3>(6));
  state_.b_g += dx.segment<3>(9);
  state_.b_a += dx.segment<3>(12);

  // Joseph form
  const Eigen::Matrix<double, 15, 15> IKH =
      Eigen::Matrix<double, 15, 15>::Identity() - K * H_all;
  state_.P = IKH * state_.P * IKH.transpose() + K * R_mat * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());

  // Enforce minimum positive-definite covariance
  if (!state_.P.allFinite()) {
    get_logger()->warn("EKF covariance contains NaN — resetting to default");
    state_.P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-2;
  }

  // ------------------------------------------------------------------
  // Posterior landmark refinement + age-based pruning.
  //
  // Re-triangulate observed landmarks using the POSTERIOR state so they
  // stay consistent with the updated estimate.  Landmarks that are NOT
  // observed this frame keep their old world positions — when they are
  // re-observed later the residual encodes multi-frame drift, which
  // provides a stronger geometric constraint than frame-to-frame.
  // ------------------------------------------------------------------
  for (const auto &f : features) {
    auto it = landmarks_.find(f.id);
    if (it != landmarks_.end() && f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
      it->second.p_w = camToWorld(f.p_c);
    }
  }

  // Prune stale landmarks (age-based sliding window)
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
    if (frame_count_ - it->second.last_seen_frame > noise_.landmark_max_age)
      it = landmarks_.erase(it);
    else
      ++it;
  }
}

// ---------------------------------------------------------------------------
// UPDATE from external 6-DOF pose (loosely-coupled VO fusion)
// ---------------------------------------------------------------------------
void EKF::updateFromPose(const Eigen::Vector3d &p_meas,
                         const Eigen::Quaterniond &q_meas,
                         double sigma_p, double sigma_q) {
  // Residual: [dp; dtheta]
  Eigen::Matrix<double, 6, 1> z;
  z.head<3>() = p_meas - state_.p;
  // Orientation error in body frame: dtheta such that R_meas = R_state * exp([dtheta]x)
  z.tail<3>() = logSO3(state_.q.toRotationMatrix().transpose() *
                        q_meas.toRotationMatrix());

  if (!z.allFinite()) {
    get_logger()->warn("EKF pose update: residual contains NaN — skipping");
    return;
  }

  // Jacobian H (6x15): identity for position (idx 0:2) and orientation (idx 6:8)
  Eigen::Matrix<double, 6, 15> H;
  H.setZero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // dp/ddp
  H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();  // dtheta/ddtheta

  // Measurement noise
  Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
  R.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);
  R.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_q * sigma_q);

  // Mahalanobis gating (chi-squared 6-DOF, 95% = 12.59)
  const Eigen::Matrix<double, 6, 6> S = H * state_.P * H.transpose() + R;
  const double mahal = z.transpose() * S.inverse() * z;
  if (mahal > 12.59) return;

  // Kalman update
  const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> S_ldlt(S);
  if (S_ldlt.info() != Eigen::Success) {
    get_logger()->warn("EKF pose update: LDLT decomposition failed");
    return;
  }
  const Eigen::Matrix<double, 15, 6> K =
      state_.P * H.transpose() * S_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
  const Eigen::Matrix<double, 15, 1> dx = K * z;

  if (!dx.allFinite()) {
    get_logger()->warn("EKF pose update: dx contains NaN — skipping");
    return;
  }

  state_.p += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.q = boxplus(state_.q, dx.segment<3>(6));
  state_.b_g += dx.segment<3>(9);
  state_.b_a += dx.segment<3>(12);

  // Joseph form
  const Eigen::Matrix<double, 15, 15> IKH =
      Eigen::Matrix<double, 15, 15>::Identity() - K * H;
  state_.P = IKH * state_.P * IKH.transpose() + K * R * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// RK4 IMU integration
// ---------------------------------------------------------------------------
EKF::PVQ EKF::integrateRK4(const PVQ &pvq, const Eigen::Vector3d &omega_c,
                           const Eigen::Vector3d &a_c, double dt) const {
  const Eigen::Vector3d g = gravity();
  // Derivative function: given state, return (ṗ, v̇, q̇ axis-angle)
  auto deriv = [&](const PVQ &s) -> PVQ {
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
  const Eigen::Matrix3d R2 = R * expSO3(omega_c * 0.5 * dt);
  const Eigen::Vector3d dp2 = v2;
  const Eigen::Vector3d dv2 = R2 * a_c + g;

  // k3 (same mid-point rotation as k2)
  const Eigen::Vector3d v3 = pvq.v + 0.5 * dt * dv2;
  const Eigen::Vector3d dp3 = v3;
  const Eigen::Vector3d dv3 = R2 * a_c + g;

  // k4 (full step rotation)
  const Eigen::Vector3d v4 = pvq.v + dt * dv3;
  const Eigen::Matrix3d R4 = R * expSO3(omega_c * dt);
  const Eigen::Vector3d dp4 = v4;
  const Eigen::Vector3d dv4 = R4 * a_c + g;

  PVQ next;
  next.p = pvq.p + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
  next.v = pvq.v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
  next.q = Eigen::Quaterniond(R * expSO3(omega_c * dt)).normalized();
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
void EKF::computeFG(const Eigen::Vector3d &omega_c, const Eigen::Vector3d &a_c,
                    Eigen::Matrix<double, 15, 15> &F,
                    Eigen::Matrix<double, 15, 12> &G) const {
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
  G.block<3, 3>(3, 3) = -R; // accel noise → velocity
  G.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity(); // gyro noise → angle
  G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // gyro bias noise
  G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // accel bias noise
}

// ---------------------------------------------------------------------------
// Project 3-D point in left camera frame to stereo pixel pair
// ---------------------------------------------------------------------------
void EKF::project(const Eigen::Vector3d &p_c, double &u_l, double &v_l,
                  double &u_r, double &v_r) const {
  const double inv_z = 1.0 / p_c.z();
  u_l = cam_.fx * p_c.x() * inv_z + cam_.cx;
  v_l = cam_.fy * p_c.y() * inv_z + cam_.cy;
  u_r = cam_.fx * (p_c.x() - cam_.baseline) * inv_z + cam_.cx;
  v_r = v_l;
}

// ---------------------------------------------------------------------------
// Coordinate transforms: camera ↔ world
//
//   T_{cam←imu} given by cam_.T_cam_imu  →  R_ci, t_ci
//   State quaternion q  →  R_{w←b} = R_wb  (body/IMU to world)
//
//   cam→world:  p_imu = R_ci^T * (p_c - t_ci)
//               p_w   = R_wb * p_imu + p
//
//   world→cam:  p_imu = R_wb^T * (p_w - p)
//               p_c   = R_ci * p_imu + t_ci
// ---------------------------------------------------------------------------
Eigen::Vector3d EKF::camToWorld(const Eigen::Vector3d &p_c) const {
  const Eigen::Matrix3d R_wb = state_.q.toRotationMatrix();
  const Eigen::Matrix3d R_ci = cam_.T_cam_imu.rotation();
  const Eigen::Vector3d t_ci = cam_.T_cam_imu.translation();
  const Eigen::Vector3d p_imu = R_ci.transpose() * (p_c - t_ci);
  return R_wb * p_imu + state_.p;
}

Eigen::Vector3d EKF::worldToCam(const Eigen::Vector3d &p_w) const {
  const Eigen::Matrix3d R_wb = state_.q.toRotationMatrix();
  const Eigen::Matrix3d R_ci = cam_.T_cam_imu.rotation();
  const Eigen::Vector3d t_ci = cam_.T_cam_imu.translation();
  const Eigen::Vector3d p_imu = R_wb.transpose() * (p_w - state_.p);
  return R_ci * p_imu + t_ci;
}

} // namespace ekf_vio
