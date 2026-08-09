// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/ekf.hpp"

#include <algorithm>
#include <cmath>
#include <ekf_vio/logging.hpp>
#include <ekf_vio/math_utils.hpp>
#include <numeric>
#include <utility>

namespace ekf_vio {

using namespace math;

// ---------------------------------------------------------------------------
EKF::EKF(StereoCamera cam, const NoiseParams& noise) : cam_(std::move(cam)), noise_(noise) {
  // G * Q_c * G^T is rotation-independent: every G block is ±I or ±R, and
  // R*R^T = I, so the product reduces to a constant diagonal-block matrix.
  //
  //  State rows     G slice          Contribution
  //  [3:6,  3:6]   -R * σ_a         σ_a² * R*R^T = σ_a² * I
  //  [6:9,  6:9]   -I  * σ_g        σ_g² * I
  //  [9:12, 9:12]   I  * σ_gb       σ_gb² * I
  //  [12:15,12:15]  I  * σ_ab       σ_ab² * I
  gqgt_.setZero();
  gqgt_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (noise_.sigma_accel * noise_.sigma_accel);
  gqgt_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (noise_.sigma_gyro * noise_.sigma_gyro);
  gqgt_.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_gyro_bias * noise_.sigma_gyro_bias);
  gqgt_.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_accel_bias * noise_.sigma_accel_bias);
}

// ---------------------------------------------------------------------------
// PREDICT step
// ---------------------------------------------------------------------------
void EKF::predict(const ImuData& imu, double dt) {
  // 1. Bias-corrected measurements
  const Eigen::Vector3d omega_c = imu.gyro - state_.b_g;
  const Eigen::Vector3d a_c = imu.accel - state_.b_a;

  // 2. Integrate state (RK4)
  PVQ pvq{.p = state_.T_wb.translation(), .v = state_.v, .R = state_.T_wb.so3()};
  pvq = integrateRK4(pvq, omega_c, a_c, dt);
  state_.T_wb = Sophus::SE3d(pvq.R, pvq.p);
  state_.v = pvq.v;

  // 3. Compute error-state Jacobian F (15×15)
  Eigen::Matrix<double, 15, 15> F;
  computeF(omega_c, a_c, F);

  // 4. Discretise  Phi ≈ I + F*dt,  Q_d = gqgt_ * dt  (gqgt_ is constant, precomputed)
  const Eigen::Matrix<double, 15, 15> Phi = Eigen::Matrix<double, 15, 15>::Identity() + F * dt;
  const Eigen::Matrix<double, 15, 15> Q_d = gqgt_ * dt;

  // 5. Propagate covariance  P ← Phi P Phi^T + Q_d
  state_.P = Phi * state_.P * Phi.transpose() + Q_d;

  // Symmetrise (numerical insurance)
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// UPDATE step — stereo reprojection against landmark map
// ---------------------------------------------------------------------------
void EKF::update(const std::vector<Feature>& features) {
  if (features.empty()) return;

  ++frame_count_;

  // Precompute transforms
  const Eigen::Matrix3d R_wb = state_.T_wb.rotationMatrix();     // body→world
  const Eigen::Matrix3d R_ci = cam_.T_cam_imu.rotationMatrix();  // imu→cam
  // World→camera: R_cw = R_ci * R_wb^T
  const Eigen::Matrix3d R_cw = R_ci * R_wb.transpose();

  get_logger()->debug("[EKF upd] frame={} features_in={}  landmarks_map={}", frame_count_,
                      features.size(), landmarks_.size());

  // ------------------------------------------------------------------
  // Separate features into new (initialise landmark) vs tracked (measure)
  // ------------------------------------------------------------------
  std::vector<int> meas_indices;  // indices into features[] for measurement
  int n_new_landmarks = 0;
  for (int i = 0; i < static_cast<int>(features.size()); ++i) {
    const Feature& f = features[i];
    auto it = landmarks_.find(f.id);
    if (it == landmarks_.end()) {
      // New landmark: triangulate and store in world frame
      if (f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
        landmarks_[f.id] = {.p_w = camToWorld(f.p_c), .last_seen_frame = frame_count_};
        ++n_new_landmarks;
      }
    } else {
      it->second.last_seen_frame = frame_count_;
      meas_indices.push_back(i);
    }
  }

  const int M = static_cast<int>(meas_indices.size());
  get_logger()->debug("[EKF upd]   new_landmarks={}  meas_candidates={}", n_new_landmarks, M);
  if (M == 0) return;

  // ------------------------------------------------------------------
  // Build stacked residuals z and Jacobians H, with outlier rejection
  // ------------------------------------------------------------------
  const double sig2 = noise_.sigma_pixel * noise_.sigma_pixel;
  // Chi-squared threshold for 3-DOF at 95% confidence.
  // Measurement is z=[u_l,v_l,u_r]: for rectified stereo, v_r≡v_l both in the
  // observation model and (under ZNCC) in the tracker output, so a 4th residual
  // on v_r would double-count the vertical constraint and over-weight pitch.
  const double chi2_thresh = 7.815;

  // Accepted feature indices + prior residuals (for capping only).
  // H and innovations are recomputed at the current state in the sequential loop.
  std::vector<int> accepted_idx;
  std::vector<Eigen::Vector3d> residuals;
  accepted_idx.reserve(M);
  residuals.reserve(M);

  int n_behind_camera = 0;
  int n_pixel_gate_fail = 0;
  int n_mahal_fail = 0;

  for (int k = 0; k < M; ++k) {
    const Feature& f = features[meas_indices[k]];
    const Eigen::Vector3d& p_w = landmarks_.at(f.id).p_w;

    // Predict camera-frame point from world landmark + current state
    const Eigen::Vector3d p_c_pred = worldToCam(p_w);

    // Skip landmarks behind the camera or too far
    if (p_c_pred.z() < 0.1 || p_c_pred.z() > 50.0) {
      ++n_behind_camera;
      continue;
    }

    // Predicted stereo projection
    double eu_l = 0.0;
    double ev_l = 0.0;
    double eu_r = 0.0;
    double ev_r = 0.0;
    project(p_c_pred, eu_l, ev_l, eu_r, ev_r);

    // Residual  z - h(x)  with z = [u_l, v_l, u_r]
    Eigen::Vector3d res;
    res(0) = f.u_l - eu_l;
    res(1) = f.v_l - ev_l;
    res(2) = f.u_r - eu_r;

    // --- Measurement Jacobian (3×15) for Mahalanobis gating ---
    const double z_c = p_c_pred.z();
    const double z_c2 = z_c * z_c;

    // ∂[u_l,v_l]/∂p_c
    Eigen::Matrix<double, 2, 3> J_l;
    J_l << cam_.fx / z_c, 0.0, -cam_.fx * p_c_pred.x() / z_c2, 0.0, cam_.fy / z_c,
        -cam_.fy * p_c_pred.y() / z_c2;

    // ∂u_r/∂p_c  (right camera horizontal only; v_r is not an independent meas)
    Eigen::Matrix<double, 1, 3> J_ur;
    J_ur << cam_.fx / z_c, 0.0, -cam_.fx * (p_c_pred.x() - cam_.baseline) / z_c2;

    // p_c = R_ci * R_wb^T * (p_w - p) + t_ci
    // ∂p_c/∂δp = -R_cw = -R_ci * R_wb^T
    const Eigen::Matrix3d dp_c_dp = -R_cw;

    // ∂p_c/∂δθ: perturb R_wb → R_wb * exp([δθ]×), then R_wb^T → exp(-[δθ]×)*R_wb^T
    //   δp_c = R_ci * [R_wb^T * (p_w - p)]× * δθ
    const Eigen::Vector3d p_imu = R_wb.transpose() * (p_w - state_.T_wb.translation());
    const Eigen::Matrix3d dp_c_dtheta = R_ci * skew(p_imu);

    // Stack into H (3×15)
    Eigen::Matrix<double, 3, 15> H_i;
    H_i.setZero();
    H_i.block<2, 3>(0, 0) = J_l * dp_c_dp;   // position → left
    H_i.block<1, 3>(2, 0) = J_ur * dp_c_dp;  // position → right u
    H_i.block<2, 3>(0, 6) = J_l * dp_c_dtheta;   // orientation → left
    H_i.block<1, 3>(2, 6) = J_ur * dp_c_dtheta;  // orientation → right u

    // Pixel-space hard gate — independent of P size.
    // When P is large the innovation covariance S = H P Hᵀ + R is also
    // large, so the Mahalanobis distance of even a 100-px residual can
    // normalise to near zero and pass the chi² threshold.  A per-component
    // pixel cap catches bad measurements regardless of covariance state.
    constexpr double kMaxResidualPx = 40.0;
    if (res.cwiseAbs().maxCoeff() > kMaxResidualPx) {
      ++n_pixel_gate_fail;
      get_logger()->debug("[EKF upd]   pixel gate reject: res=[{:.1f},{:.1f},{:.1f}]", res(0),
                          res(1), res(2));
      continue;
    }

    // Mahalanobis gating: reject outliers using innovation covariance
    const Eigen::Matrix3d R_i = Eigen::Matrix3d::Identity() * sig2;
    const Eigen::Matrix3d S_i = H_i * state_.P * H_i.transpose() + R_i;
    const double mahal = res.transpose() * S_i.inverse() * res;
    if (mahal > chi2_thresh) {
      ++n_mahal_fail;
      continue;
    }

    accepted_idx.push_back(meas_indices[k]);
    residuals.push_back(res);
  }

  // Cap to avoid oversized update loops (keep top features by residual norm)
  const int max_meas = 200;
  if (static_cast<int>(accepted_idx.size()) > max_meas) {
    // Keep features with smallest residuals (best matches)
    std::vector<int> idx(accepted_idx.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + max_meas, idx.end(),
                      [&](int a, int b) { return residuals[a].norm() < residuals[b].norm(); });
    std::vector<int> a2;
    a2.reserve(max_meas);
    for (int i = 0; i < max_meas; ++i) {
      a2.push_back(accepted_idx[idx[i]]);
    }
    accepted_idx = std::move(a2);
  }

  get_logger()->debug(
      "[EKF upd]   gating: behind_cam={}  pixel_fail={}  mahal_fail={}  accepted={}",
      n_behind_camera, n_pixel_gate_fail, n_mahal_fail, accepted_idx.size());

  const auto N = static_cast<Eigen::Index>(accepted_idx.size());
  if (N == 0) return;

  // ------------------------------------------------------------------
  // Sequential Kalman update
  //
  // Batch update requires LDLT on a (3N × 3N) matrix — O((3N)^3) FLOP.
  // Sequential update processes each 3-DOF feature measurement
  // independently.  Each step only needs a 3×3 Cholesky, so the total
  // cost is O(N × 15²).
  //
  // CRITICAL — recompute residual and H at the current state after each
  // feature.  Freezing innovations at the prior and applying them with
  // shrinking P over-corrects (later features still see the full prior
  // error).  That is NOT equivalent to the batch update.
  // ------------------------------------------------------------------
  const Eigen::Matrix3d R_i = Eigen::Matrix3d::Identity() * sig2;

  get_logger()->debug("[EKF upd]   sequential update: N={}  P_trace_prior={:.4e}", N,
                      state_.P.trace());

  // Accumulate total correction for post-update diagnostics only
  Eigen::Matrix<double, 15, 1> dx_total = Eigen::Matrix<double, 15, 1>::Zero();

  for (Eigen::Index k = 0; k < N; ++k) {
    const Feature& f = features[accepted_idx[static_cast<size_t>(k)]];
    const Eigen::Vector3d& p_w = landmarks_.at(f.id).p_w;

    // Re-linearise observation model at the current (partially updated) state
    const Eigen::Matrix3d R_wb_k = state_.T_wb.rotationMatrix();
    const Eigen::Matrix3d R_cw_k = R_ci * R_wb_k.transpose();
    const Eigen::Vector3d p_c_pred = worldToCam(p_w);
    if (p_c_pred.z() < 0.1 || p_c_pred.z() > 50.0) continue;

    double eu_l = 0.0;
    double ev_l = 0.0;
    double eu_r = 0.0;
    double ev_r = 0.0;
    project(p_c_pred, eu_l, ev_l, eu_r, ev_r);

    Eigen::Vector3d res;
    res(0) = f.u_l - eu_l;
    res(1) = f.v_l - ev_l;
    res(2) = f.u_r - eu_r;

    const double z_c = p_c_pred.z();
    const double z_c2 = z_c * z_c;
    Eigen::Matrix<double, 2, 3> J_l;
    J_l << cam_.fx / z_c, 0.0, -cam_.fx * p_c_pred.x() / z_c2, 0.0, cam_.fy / z_c,
        -cam_.fy * p_c_pred.y() / z_c2;
    Eigen::Matrix<double, 1, 3> J_ur;
    J_ur << cam_.fx / z_c, 0.0, -cam_.fx * (p_c_pred.x() - cam_.baseline) / z_c2;

    const Eigen::Matrix3d dp_c_dp = -R_cw_k;
    const Eigen::Vector3d p_imu = R_wb_k.transpose() * (p_w - state_.T_wb.translation());
    const Eigen::Matrix3d dp_c_dtheta = R_ci * skew(p_imu);

    Eigen::Matrix<double, 3, 15> H_k;
    H_k.setZero();
    H_k.block<2, 3>(0, 0) = J_l * dp_c_dp;
    H_k.block<1, 3>(2, 0) = J_ur * dp_c_dp;
    H_k.block<2, 3>(0, 6) = J_l * dp_c_dtheta;
    H_k.block<1, 3>(2, 6) = J_ur * dp_c_dtheta;

    // Innovation covariance (3×3) — Cholesky is stable and trivially cheap
    const Eigen::Matrix3d S_k = H_k * state_.P * H_k.transpose() + R_i;
    const Eigen::LLT<Eigen::Matrix3d> S_llt(S_k);
    if (S_llt.info() != Eigen::Success) continue;

    // Kalman gain (15×3)
    const Eigen::Matrix<double, 15, 3> K_k =
        state_.P * H_k.transpose() * S_llt.solve(Eigen::Matrix3d::Identity());

    const Eigen::Matrix<double, 15, 1> dx_k = K_k * res;
    if (!dx_k.allFinite()) continue;

    dx_total += dx_k;

    state_.T_wb.translation() += dx_k.segment<3>(0);
    state_.v += dx_k.segment<3>(3);
    state_.T_wb.so3() *= Sophus::SO3d::exp(dx_k.segment<3>(6));
    state_.b_g += dx_k.segment<3>(9);
    state_.b_a += dx_k.segment<3>(12);

    // Covariance update — Joseph form for numerical stability
    const Eigen::Matrix<double, 15, 15> IKH = Eigen::Matrix<double, 15, 15>::Identity() - K_k * H_k;
    state_.P = IKH * state_.P * IKH.transpose() + K_k * R_i * K_k.transpose();
    state_.P = 0.5 * (state_.P + state_.P.transpose());
  }

  get_logger()->debug(
      "[EKF upd]   P_trace_post={:.4e}  dx: pos={:.4f}m  vel={:.4f}m/s  rot={:.4f}rad"
      "  bg={:.2e}  ba={:.2e}",
      state_.P.trace(), dx_total.segment<3>(0).norm(), dx_total.segment<3>(3).norm(),
      dx_total.segment<3>(6).norm(), dx_total.segment<3>(9).norm(), dx_total.segment<3>(12).norm());

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
  for (const auto& f : features) {
    auto it = landmarks_.find(f.id);
    if (it != landmarks_.end() && f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
      it->second.p_w = camToWorld(f.p_c);
    }
  }

  // Prune stale landmarks (age-based sliding window)
  for (auto it = landmarks_.begin(); it != landmarks_.end();) {
    if (frame_count_ - it->second.last_seen_frame > noise_.landmark_max_age) {
      it = landmarks_.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
// UPDATE from external 6-DOF pose (loosely-coupled VO fusion)
// ---------------------------------------------------------------------------
void EKF::updateFromPose(const Sophus::SE3d& T_meas, double sigma_p, double sigma_q) {
  // Residual: [dp; dtheta]
  Eigen::Matrix<double, 6, 1> z;
  z.head<3>() = T_meas.translation() - state_.T_wb.translation();
  // Orientation error in body frame: dtheta such that R_meas = R_state * exp([dtheta]x)
  z.tail<3>() = (state_.T_wb.so3().inverse() * T_meas.so3()).log();

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

  state_.T_wb.translation() += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.T_wb.so3() *= Sophus::SO3d::exp(dx.segment<3>(6));
  state_.b_g += dx.segment<3>(9);
  state_.b_a += dx.segment<3>(12);

  // Joseph form
  const Eigen::Matrix<double, 15, 15> IKH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
  state_.P = IKH * state_.P * IKH.transpose() + K * R * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// RK4 IMU integration
// ---------------------------------------------------------------------------
EKF::PVQ EKF::integrateRK4(const PVQ& pvq, const Eigen::Vector3d& omega_c,
                           const Eigen::Vector3d& a_c, double dt) const {
  const Eigen::Vector3d g = gravity();
  // k1
  const Eigen::Vector3d dp1 = pvq.v;
  const Eigen::Vector3d dv1 = pvq.R.matrix() * a_c + g;

  // k2 (use mid-point velocity from k1)
  // we move dt / 2 forward using the slopes from k1
  // predict vel at t + (dt / 2) = v_t + 0.5 * dt * dv1
  const Eigen::Vector3d v2 = pvq.v + 0.5 * dt * dv1;
  // predict rot at t + (dt / 2) = R_t * Exp(omega_c * 0.5 * t)
  const Eigen::Matrix3d R2 = (pvq.R * Sophus::SO3d::exp(omega_c * 0.5 * dt)).matrix();
  // new pos in the slope using predicted velocity
  const Eigen::Vector3d& dp2 = v2;
  // new vel in the slop using predict rotation
  const Eigen::Vector3d dv2 = R2 * a_c + g;

  // k3 (same mid-point rotation as k2)
  // we go back to the start and love dt / 2 forward again but using better slopes from k2
  // A refined vel predicted for mid-point
  const Eigen::Vector3d v3 = pvq.v + 0.5 * dt * dv2;
  const Eigen::Vector3d& dp3 = v3;
  // assume omega_c is constant R2 is reused
  const Eigen::Vector3d dv3 = R2 * a_c + g;

  // k4 (full step rotation)
  // move a full dt forward using the slopes from k3
  const Eigen::Vector3d v4 = pvq.v + dt * dv3;
  const Eigen::Matrix3d R4 = (pvq.R * Sophus::SO3d::exp(omega_c * dt)).matrix();
  const Eigen::Vector3d& dp4 = v4;
  const Eigen::Vector3d dv4 = R4 * a_c + g;

  PVQ next;
  next.p = pvq.p + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
  next.v = pvq.v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
  next.R = pvq.R * Sophus::SO3d::exp(omega_c * dt);
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
void EKF::computeF(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                   Eigen::Matrix<double, 15, 15>& F) const {
  F.setZero();

  const Eigen::Matrix3d R = state_.T_wb.rotationMatrix();

  // ṗ = v
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  // v̇ = R*a_c + g  (wrt error state)
  F.block<3, 3>(3, 6) = -R * skew(a_c);  // ∂/∂δθ
  F.block<3, 3>(3, 12) = -R;             // ∂/∂δb_a
  // θ̇ = ω_c
  F.block<3, 3>(6, 6) = -skew(omega_c);                // ∂/∂δθ
  F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity();  // ∂/∂δb_g
}

// ---------------------------------------------------------------------------
// Project 3-D point in left camera frame to stereo pixel pair
// ---------------------------------------------------------------------------
void EKF::project(const Eigen::Vector3d& p_c, double& u_l, double& v_l, double& u_r,
                  double& v_r) const {
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
Eigen::Vector3d EKF::camToWorld(const Eigen::Vector3d& p_c) const {
  return state_.T_wb * (cam_.T_cam_imu.inverse() * p_c);
}

Eigen::Vector3d EKF::worldToCam(const Eigen::Vector3d& p_w) const {
  return cam_.T_cam_imu * (state_.T_wb.inverse() * p_w);
}

}  // namespace ekf_vio
