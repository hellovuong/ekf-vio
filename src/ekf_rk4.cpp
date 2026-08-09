// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/ekf_rk4.hpp"

#include <algorithm>
#include <cmath>
#include <ekf_vio/logging.hpp>
#include <ekf_vio/math_utils.hpp>
#include <numeric>
#include <utility>

namespace ekf_vio {

using namespace math;

// ---------------------------------------------------------------------------
EKFRk4::EKFRk4(StereoCamera cam, const NoiseParams& noise) : cam_(std::move(cam)), noise_(noise) {
  // Same precomputed GQG^T as EKF — rotation-invariant, computed once.
  gqgt_.setZero();
  gqgt_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (noise_.sigma_accel * noise_.sigma_accel);
  gqgt_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (noise_.sigma_gyro * noise_.sigma_gyro);
  gqgt_.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_gyro_bias * noise_.sigma_gyro_bias);
  gqgt_.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * (noise_.sigma_accel_bias * noise_.sigma_accel_bias);
}

// ---------------------------------------------------------------------------
// PREDICT — full RK4 on (state, Φ, Q_d) with IMU midpoint interpolation
// ---------------------------------------------------------------------------
void EKFRk4::predict(const ImuData& imu, double dt) {
  // ── Bias-corrected start / end / midpoint IMU readings ──────────────────
  // On the first call prev_imu_ is unset; fall back to ZOH (same as EKF).
  const Eigen::Vector3d omega_start = (has_prev_imu_ ? prev_imu_.gyro : imu.gyro) - state_.b_g;
  const Eigen::Vector3d a_start = (has_prev_imu_ ? prev_imu_.accel : imu.accel) - state_.b_a;
  const Eigen::Vector3d omega_end = imu.gyro - state_.b_g;
  const Eigen::Vector3d a_end = imu.accel - state_.b_a;
  const Eigen::Vector3d omega_mid = 0.5 * (omega_start + omega_end);
  const Eigen::Vector3d a_mid = 0.5 * (a_start + a_end);

  // ── Initial state ────────────────────────────────────────────────────────
  const PVQ pvq0{.p = state_.T_wb.translation(), .v = state_.v, .R = state_.T_wb.so3()};

  // Exact midpoint and end rotations via SO3 exp (Lie group integration).
  // These are fixed for all stages — omega is the gyro reading at that point.
  const Sophus::SO3d R_mid = pvq0.R * Sophus::SO3d::exp(omega_mid * 0.5 * dt);
  const Sophus::SO3d R_end = pvq0.R * Sophus::SO3d::exp(omega_end * dt);

  // ── RK4 initial conditions for matrix ODEs ───────────────────────────────
  //   Φ(0) = I  →  Φ(dt) = state transition matrix for this step
  //   P(0) = 0  →  P(dt) = discrete-time process noise Q_d for this step
  const Eigen::Matrix<double, 15, 15> Phi0 = Eigen::Matrix<double, 15, 15>::Identity();
  const Eigen::Matrix<double, 15, 15> P0 = Eigen::Matrix<double, 15, 15>::Zero();

  // ── k1: start of interval ────────────────────────────────────────────────
  const Deriv d1 = evalDeriv(omega_start, a_start, pvq0, Phi0, P0);

  // ── k2: midpoint, first estimate ─────────────────────────────────────────
  PVQ pvq2;
  pvq2.p = pvq0.p + 0.5 * dt * d1.dp;
  pvq2.v = pvq0.v + 0.5 * dt * d1.dv;
  pvq2.R = R_mid;
  const Deriv d2 =
      evalDeriv(omega_mid, a_mid, pvq2, Phi0 + 0.5 * dt * d1.dPhi, P0 + 0.5 * dt * d1.dP);

  // ── k3: midpoint, refined estimate ───────────────────────────────────────
  PVQ pvq3;
  pvq3.p = pvq0.p + 0.5 * dt * d2.dp;
  pvq3.v = pvq0.v + 0.5 * dt * d2.dv;
  pvq3.R = R_mid;  // same midpoint rotation for k3
  const Deriv d3 =
      evalDeriv(omega_mid, a_mid, pvq3, Phi0 + 0.5 * dt * d2.dPhi, P0 + 0.5 * dt * d2.dP);

  // ── k4: end of interval ──────────────────────────────────────────────────
  PVQ pvq4;
  pvq4.p = pvq0.p + dt * d3.dp;
  pvq4.v = pvq0.v + dt * d3.dv;
  pvq4.R = R_end;
  const Deriv d4 = evalDeriv(omega_end, a_end, pvq4, Phi0 + dt * d3.dPhi, P0 + dt * d3.dP);

  // ── Weighted sum (RK4 formula) ────────────────────────────────────────────
  constexpr double k1_6 = 1.0 / 6.0;
  const Eigen::Matrix<double, 15, 15> dPhi_sum = d1.dPhi + 2.0 * d2.dPhi + 2.0 * d3.dPhi + d4.dPhi;
  const Eigen::Matrix<double, 15, 15> dP_sum = d1.dP + 2.0 * d2.dP + 2.0 * d3.dP + d4.dP;

  // ── Apply state update ────────────────────────────────────────────────────
  state_.T_wb.translation() = pvq0.p + dt * k1_6 * (d1.dp + 2.0 * d2.dp + 2.0 * d3.dp + d4.dp);
  state_.v = pvq0.v + dt * k1_6 * (d1.dv + 2.0 * d2.dv + 2.0 * d3.dv + d4.dv);
  state_.T_wb.so3() = R_end;  // exact rotation via SO3::exp(omega_end * dt)

  // ── Apply covariance update ───────────────────────────────────────────────
  //   Φ  = I + (dt/6) · ΣdΦ        (O(dt⁵) accurate)
  //   Q_d =    (dt/6) · ΣdP        (O(dt⁵) accurate)
  //   P_new = Φ · P_old · Φ^T + Q_d
  const Eigen::Matrix<double, 15, 15> Phi = Phi0 + dt * k1_6 * dPhi_sum;
  const Eigen::Matrix<double, 15, 15> Q_d = dt * k1_6 * dP_sum;

  state_.P = Phi * state_.P * Phi.transpose() + Q_d;
  state_.P = 0.5 * (state_.P + state_.P.transpose());

  // ── Buffer reading for next step's start-of-interval ────────────────────
  prev_imu_ = imu;
  has_prev_imu_ = true;
}

// ---------------------------------------------------------------------------
// evalDeriv — one RK4 stage evaluation
// ---------------------------------------------------------------------------
EKFRk4::Deriv EKFRk4::evalDeriv(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                                const PVQ& pvq, const Eigen::Matrix<double, 15, 15>& Phi,
                                const Eigen::Matrix<double, 15, 15>& P) const {
  Deriv d;

  // State derivatives
  d.dp = pvq.v;
  d.dv = pvq.R.matrix() * a_c + gravity();

  // Continuous-time Jacobian F at this (omega, a, R)
  Eigen::Matrix<double, 15, 15> F;
  computeF(omega_c, a_c, pvq.R.matrix(), F);

  // Matrix ODE derivatives
  d.dPhi = F * Phi;
  d.dP = F * P + P * F.transpose() + gqgt_;

  return d;
}

// ---------------------------------------------------------------------------
// computeF — continuous-time error-state Jacobian (15×15)
//
// State order: p(0:3), v(3:6), θ(6:9), b_g(9:12), b_a(12:15)
//   ṗ  = v                 → F[0:3, 3:6]  = I
//   v̇  = R·a + g           → F[3:6, 6:9]  = −R·[a]×
//                            F[3:6,12:15] = −R
//   θ̇  = ω                 → F[6:9, 6:9]  = −[ω]×
//                            F[6:9, 9:12] = −I
// ---------------------------------------------------------------------------
void EKFRk4::computeF(const Eigen::Vector3d& omega_c, const Eigen::Vector3d& a_c,
                      const Eigen::Matrix3d& R, Eigen::Matrix<double, 15, 15>& F) const {
  F.setZero();
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(3, 6) = -R * skew(a_c);
  F.block<3, 3>(3, 12) = -R;
  F.block<3, 3>(6, 6) = -skew(omega_c);
  F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity();
}

// ---------------------------------------------------------------------------
// UPDATE — EKF measurement update from triangulated stereo features
//
// Overview
// --------
// Each stereo feature produces a 3-DOF measurement:
//   z_i = [u_l, v_l, u_r]^T   (rectified stereo: v_r ≡ v_l, not independent)
//
// The predicted measurement is obtained by projecting the stored world-frame
// landmark p_w through the current pose estimate:
//   p_c = T_cam_imu · T_wb^{-1} · p_w          (world → body → camera)
//   ẑ_i = π(p_c)  with  π(·) = pinhole + baseline shift for right image
//
// The residual (innovation):
//   r_i = z_i − ẑ_i ∈ ℝ³
//
// Landmark management
// -------------------
// Features seen for the first time are initialised as landmarks (p_w from
// stereo triangulation) and skipped for this update step — we need at least
// one prior observation before we can correct the state.  Features seen
// before produce a measurement residual and update the state.
//
// Gating
// -------
// Three rejection stages before a measurement enters the Kalman update:
//   1. Depth gate    : 0.1 m < z_c < 50 m  (numerical safety for projection)
//   2. Pixel gate    : |r_i|_∞ < 40 px     (fast rejection of gross outliers)
//   3. Mahalanobis   : r_i^T S_i^{-1} r_i < χ²(3, 0.95) = 7.815
//                      where S_i = H_i P H_i^T + R_i   (innovation covariance)
//      This is a chi-squared test with 3 DOF per landmark.
//
// Sequential (iterated) Kalman update
// ------------------------------------
// Measurements are fused one at a time (sequential EKF update), which keeps
// each individual update matrix small (3×3 inverse instead of 3N×3N) and
// preserves positive-definiteness of P more easily.
//
// Joseph-form covariance update for numerical stability:
//   K_k  = P H_k^T S_k^{-1}                   (Kalman gain, 15×3)
//   dx_k = K_k r_k                              (state correction, 15×1)
//   P    = (I − K_k H_k) P (I − K_k H_k)^T + K_k R K_k^T
//        = IKH · P · IKH^T + K_k R K_k^T       (Joseph form — always PSD)
//
// State correction on the manifold:
//   p    ← p    + dx[0:3]
//   v    ← v    + dx[3:6]
//   R_wb ← R_wb · Exp(dx[6:9])    (right-multiply SO3 update)
//   b_g  ← b_g  + dx[9:12]
//   b_a  ← b_a  + dx[12:15]
// ---------------------------------------------------------------------------
void EKFRk4::update(const std::vector<Feature>& features) {
  if (features.empty()) return;

  ++frame_count_;

  // Rotation matrices needed throughout:
  //   R_wb = rotation world←body (from current pose estimate)
  //   R_ci = rotation cam←imu    (fixed extrinsic, from calibration)
  //   R_cw = R_ci · R_wb^T       = rotation cam←world (for dp_c/dp and dp_c/dθ)
  const Eigen::Matrix3d R_wb = state_.T_wb.rotationMatrix();
  const Eigen::Matrix3d R_ci = cam_.T_cam_imu.rotationMatrix();
  const Eigen::Matrix3d R_cw = R_ci * R_wb.transpose();

  get_logger()->debug("[EKFRk4 upd] frame={} features_in={}  landmarks_map={}", frame_count_,
                      features.size(), landmarks_.size());

  // ── Landmark management ────────────────────────────────────────────────────
  // Split incoming features into:
  //   • New landmarks  → initialise p_w from triangulated p_c; skip update this frame.
  //   • Known landmarks → produce a measurement residual; collect index for update.
  std::vector<int> meas_indices;
  int n_new_landmarks = 0;
  for (int i = 0; i < static_cast<int>(features.size()); ++i) {
    const Feature& f = features[i];
    auto it = landmarks_.find(f.id);
    if (it == landmarks_.end()) {
      // First observation: initialise world-frame position from triangulation.
      // p_w = T_wb · T_cam_imu^{-1} · p_c
      if (f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
        landmarks_[f.id] = {.p_w = camToWorld(f.p_c), .last_seen_frame = frame_count_};
        ++n_new_landmarks;
      }
    } else {
      // Known landmark: mark it alive and queue for measurement update.
      it->second.last_seen_frame = frame_count_;
      meas_indices.push_back(i);
    }
  }

  const int M = static_cast<int>(meas_indices.size());
  get_logger()->debug("[EKFRk4 upd]   new_landmarks={}  meas_candidates={}", n_new_landmarks, M);
  if (meas_indices.empty()) return;

  // σ² = pixel noise variance; used in measurement covariance R_i = σ²·I₃
  const double sig2 = noise_.sigma_pixel * noise_.sigma_pixel;

  // χ²(3 DOF, 95th percentile) = 7.815 — Mahalanobis gating threshold.
  // z=[u_l,v_l,u_r] only: v_r is not independent under rectified stereo / ZNCC.
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

  // ── Build residuals and Jacobians, with gating ─────────────────────────────
  for (int k = 0; k < M; ++k) {
    const Feature& f = features[meas_indices[k]];
    const Eigen::Vector3d& p_w = landmarks_.at(f.id).p_w;

    // Project landmark into camera frame using current pose estimate.
    // p_c = T_cam_imu · T_wb^{-1} · p_w
    const Eigen::Vector3d p_c_pred = worldToCam(p_w);

    // Gate 1 — depth: discard points behind or too far from camera.
    if (p_c_pred.z() < 0.1 || p_c_pred.z() > 50.0) {
      ++n_behind_camera;
      continue;
    }

    // Predicted pixel observations via pinhole projection:
    //   left  : û_l = fx · x_c/z_c + cx,   v̂_l = fy · y_c/z_c + cy
    //   right : û_r = fx · (x_c − b)/z_c + cx   (baseline shift; v̂_r = v̂_l)
    double eu_l = 0.0;
    double ev_l = 0.0;
    double eu_r = 0.0;
    double ev_r = 0.0;
    project(p_c_pred, eu_l, ev_l, eu_r, ev_r);

    // Innovation (residual) r_i = z_i − ẑ_i ∈ ℝ³ , z = [u_l, v_l, u_r]
    Eigen::Vector3d res;
    res(0) = f.u_l - eu_l;
    res(1) = f.v_l - ev_l;
    res(2) = f.u_r - eu_r;

    // ── Measurement Jacobian H_i (3×15) ───────────────────────────────────
    // The observation model z = π(p_c(x)) chains two Jacobians:
    //
    //   ∂z/∂x = ∂π/∂p_c · ∂p_c/∂x
    //
    // (a) Camera projection Jacobian ∂π/∂p_c:
    //   For the left image (2×3):
    //     J_l = [ fx/z_c,   0,    −fx·x_c/z_c² ]
    //           [   0,    fy/z_c, −fy·y_c/z_c² ]
    //   For right u only (1×3); v_r is not an independent measurement:
    //     J_ur = [ fx/z_c,  0,  −fx·(x_c−b)/z_c² ]
    const double z_c = p_c_pred.z();
    const double z_c2 = z_c * z_c;

    Eigen::Matrix<double, 2, 3> J_l;
    J_l << cam_.fx / z_c, 0.0, -cam_.fx * p_c_pred.x() / z_c2, 0.0, cam_.fy / z_c,
        -cam_.fy * p_c_pred.y() / z_c2;

    Eigen::Matrix<double, 1, 3> J_ur;
    J_ur << cam_.fx / z_c, 0.0, -cam_.fx * (p_c_pred.x() - cam_.baseline) / z_c2;

    // (b) Pose-to-point Jacobians ∂p_c/∂x:
    //
    //   p_c = R_cw · (p_w − t_wb)   where R_cw = R_ci · R_wb^T
    //
    //   ∂p_c/∂p  = −R_cw             (3×3, wrt body position t_wb)
    //
    //   ∂p_c/∂θ  = R_ci · [R_wb^T·(p_w − t_wb)]×
    //            = R_ci · [p_imu]×    (3×3, wrt orientation error θ)
    //   where p_imu = R_wb^T · (p_w − t_wb) is p_w expressed in the IMU/body frame.
    //   This comes from differentiating R_wb · exp(δθ) · p_imu ≈ R_wb·(p_imu + δθ×p_imu).
    const Eigen::Matrix3d dp_c_dp = -R_cw;
    const Eigen::Vector3d p_imu = R_wb.transpose() * (p_w - state_.T_wb.translation());
    const Eigen::Matrix3d dp_c_dtheta = R_ci * skew(p_imu);

    // Assemble H_i (3×15): non-zero blocks at position [0:3] and orientation [6:9]
    // Layout: [ p(0:3) | v(3:6) | θ(6:9) | b_g(9:12) | b_a(12:15) ]
    //
    //   H_i = [ J_l ·(∂p_c/∂p)   0   J_l ·(∂p_c/∂θ)   0   0 ]   ← left  (rows 0,1)
    //         [ J_ur·(∂p_c/∂p)   0   J_ur·(∂p_c/∂θ)   0   0 ]   ← right u (row 2)
    Eigen::Matrix<double, 3, 15> H_i;
    H_i.setZero();
    H_i.block<2, 3>(0, 0) = J_l * dp_c_dp;       // ∂(left  pixel)/∂p
    H_i.block<1, 3>(2, 0) = J_ur * dp_c_dp;      // ∂(right u)/∂p
    H_i.block<2, 3>(0, 6) = J_l * dp_c_dtheta;   // ∂(left  pixel)/∂θ
    H_i.block<1, 3>(2, 6) = J_ur * dp_c_dtheta;  // ∂(right u)/∂θ

    // Gate 2 — pixel magnitude: fast gross-outlier rejection before the
    // more expensive Mahalanobis test.
    constexpr double kMaxResidualPx = 40.0;
    if (res.cwiseAbs().maxCoeff() > kMaxResidualPx) {
      ++n_pixel_gate_fail;
      continue;
    }

    // Gate 3 — Mahalanobis distance: chi-squared test on the innovation.
    //   S_i = H_i · P · H_i^T + R_i     (3×3 innovation covariance)
    //   d²  = r_i^T · S_i^{-1} · r_i    (scalar Mahalanobis distance)
    //   Accept if d² < χ²(3, 0.95) = 7.815
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

  // ── Cap measurement count for runtime bound ────────────────────────────────
  // If many landmarks survive gating, keep only the 200 with smallest residual
  // norm (most confident / closest to prediction).  This bounds the O(N)
  // sequential update loop without discarding completely — just prioritises
  // well-predicted features.
  const int max_meas = 200;
  if (static_cast<int>(accepted_idx.size()) > max_meas) {
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
      "[EKFRk4 upd]   gating: behind_cam={}  pixel_fail={}  mahal_fail={}  accepted={}",
      n_behind_camera, n_pixel_gate_fail, n_mahal_fail, accepted_idx.size());

  const auto N = static_cast<Eigen::Index>(accepted_idx.size());
  if (N == 0) return;

  // Measurement noise covariance R = σ²·I₃  (isotropic pixel noise)
  const Eigen::Matrix3d R_i = Eigen::Matrix3d::Identity() * sig2;

  // ── Sequential Kalman update (one measurement at a time) ──────────────────
  // CRITICAL — recompute residual and H at the current state after each feature.
  // Freezing prior innovations and applying them with shrinking P over-corrects.
  for (Eigen::Index k = 0; k < N; ++k) {
    const Feature& f = features[accepted_idx[static_cast<size_t>(k)]];
    const Eigen::Vector3d& p_w = landmarks_.at(f.id).p_w;

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

    // Innovation covariance S_k = H_k · P · H_k^T + R   (3×3)
    const Eigen::Matrix3d S_k = H_k * state_.P * H_k.transpose() + R_i;

    // Cholesky decomposition of S_k for numerically stable solve.
    const Eigen::LLT<Eigen::Matrix3d> S_llt(S_k);
    if (S_llt.info() != Eigen::Success) continue;  // S not PD — skip (shouldn't happen post-gating)

    // Kalman gain K_k = P · H_k^T · S_k^{-1}   (15×3)
    const Eigen::Matrix<double, 15, 3> K_k =
        state_.P * H_k.transpose() * S_llt.solve(Eigen::Matrix3d::Identity());

    // State correction vector dx = K_k · r_k ∈ ℝ¹⁵
    const Eigen::Matrix<double, 15, 1> dx_k = K_k * res;
    if (!dx_k.allFinite()) continue;

    // Apply correction on the manifold:
    //   position        p  ← p  + dx[0:3]
    //   velocity        v  ← v  + dx[3:6]
    //   orientation     R  ← R  · Exp(dx[6:9])   (right-multiply on SO3)
    //   gyro bias       b_g← b_g+ dx[9:12]
    //   accel bias      b_a← b_a+ dx[12:15]
    state_.T_wb.translation() += dx_k.segment<3>(0);
    state_.v += dx_k.segment<3>(3);
    state_.T_wb.so3() *= Sophus::SO3d::exp(dx_k.segment<3>(6));
    state_.b_g += dx_k.segment<3>(9);
    state_.b_a += dx_k.segment<3>(12);

    // Joseph-form covariance update — numerically stable, guarantees symmetry
    // and positive semi-definiteness even with finite-precision arithmetic:
    //
    //   IKH = I − K_k · H_k                        (15×15)
    //   P   = IKH · P · IKH^T + K_k · R · K_k^T   (Joseph form)
    //
    // The second term K_k·R·K_k^T compensates for the asymmetry introduced by
    // floating-point errors in I − KH, ensuring P stays PSD.
    const Eigen::Matrix<double, 15, 15> IKH = Eigen::Matrix<double, 15, 15>::Identity() - K_k * H_k;
    state_.P = IKH * state_.P * IKH.transpose() + K_k * R_i * K_k.transpose();

    // Force exact symmetry: P = ½(P + P^T) to suppress numerical drift.
    state_.P = 0.5 * (state_.P + state_.P.transpose());
  }

  // Safety check — reset covariance if NaN propagated (indicates numerical instability).
  if (!state_.P.allFinite()) {
    get_logger()->warn("EKFRk4 covariance contains NaN — resetting to default");
    state_.P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-2;
  }

  // ── Landmark update: refresh world position from latest triangulation ──────
  // Re-project the triangulated p_c back to world frame using the *updated* pose.
  // This keeps the landmark map consistent with the corrected state estimate.
  for (const auto& f : features) {
    auto it = landmarks_.find(f.id);
    if (it != landmarks_.end() && f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
      it->second.p_w = camToWorld(f.p_c);
    }
  }

  // ── Landmark culling: remove stale landmarks ───────────────────────────────
  // A landmark not observed for more than `landmark_max_age` consecutive frames
  // is dropped from the map to bound memory and avoid stale constraints.
  for (auto it = landmarks_.begin(); it != landmarks_.end();) {
    if (frame_count_ - it->second.last_seen_frame > noise_.landmark_max_age) {
      it = landmarks_.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
// updateFromPose (identical to EKF::updateFromPose)
// ---------------------------------------------------------------------------
void EKFRk4::updateFromPose(const Sophus::SE3d& T_meas, double sigma_p, double sigma_q) {
  Eigen::Matrix<double, 6, 1> z;
  z.head<3>() = T_meas.translation() - state_.T_wb.translation();
  z.tail<3>() = (state_.T_wb.so3().inverse() * T_meas.so3()).log();

  if (!z.allFinite()) {
    get_logger()->warn("EKFRk4 pose update: residual contains NaN — skipping");
    return;
  }

  Eigen::Matrix<double, 6, 15> H;
  H.setZero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
  R.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);
  R.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_q * sigma_q);

  const Eigen::Matrix<double, 6, 6> S = H * state_.P * H.transpose() + R;
  if (z.transpose() * S.inverse() * z > 12.59) return;

  const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> S_ldlt(S);
  if (S_ldlt.info() != Eigen::Success) return;

  const Eigen::Matrix<double, 15, 6> K =
      state_.P * H.transpose() * S_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
  const Eigen::Matrix<double, 15, 1> dx = K * z;
  if (!dx.allFinite()) return;

  state_.T_wb.translation() += dx.segment<3>(0);
  state_.v += dx.segment<3>(3);
  state_.T_wb.so3() *= Sophus::SO3d::exp(dx.segment<3>(6));
  state_.b_g += dx.segment<3>(9);
  state_.b_a += dx.segment<3>(12);

  const Eigen::Matrix<double, 15, 15> IKH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
  state_.P = IKH * state_.P * IKH.transpose() + K * R * K.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
void EKFRk4::project(const Eigen::Vector3d& p_c, double& u_l, double& v_l, double& u_r,
                     double& v_r) const {
  const double inv_z = 1.0 / p_c.z();
  u_l = cam_.fx * p_c.x() * inv_z + cam_.cx;
  v_l = cam_.fy * p_c.y() * inv_z + cam_.cy;
  u_r = cam_.fx * (p_c.x() - cam_.baseline) * inv_z + cam_.cx;
  v_r = v_l;
}

Eigen::Vector3d EKFRk4::camToWorld(const Eigen::Vector3d& p_c) const {
  return state_.T_wb * (cam_.T_cam_imu.inverse() * p_c);
}

Eigen::Vector3d EKFRk4::worldToCam(const Eigen::Vector3d& p_w) const {
  return cam_.T_cam_imu * (state_.T_wb.inverse() * p_w);
}

}  // namespace ekf_vio
