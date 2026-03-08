// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_vo.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <ekf_vio/logging.hpp>
#include <random>
#include <utility>

namespace ekf_vio {

// ---------------------------------------------------------------------------
StereoVO::StereoVO(StereoCamera cam) : cam_(std::move(cam)) {}

StereoVO::StereoVO(StereoCamera cam, const Params& params)
    : cam_(std::move(cam)), params_(params) {}

// ---------------------------------------------------------------------------
int StereoVO::numKeyframeLandmarks() const {
  return static_cast<int>(keyframe_.landmarks.size());
}

// ---------------------------------------------------------------------------
Sophus::SE3d StereoVO::process(const std::vector<Feature>& features) {
  if (features.empty()) return T_wc_;

  // ------------------------------------------------------------------
  // First frame: create initial keyframe
  // ------------------------------------------------------------------
  if (!has_keyframe_) {
    createKeyframe(features);
    return T_wc_;
  }

  // ------------------------------------------------------------------
  // Match current features against keyframe landmarks (by ID)
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> pts_kf;    // 3D in keyframe camera frame
  std::vector<Eigen::Vector3d> pts_curr;  // 3D in current camera frame
  std::vector<cv::Point2f> pts_2d;        // 2D in current left image
  int tracked_from_kf = 0;

  for (const auto& f : features) {
    auto it = keyframe_.landmarks.find(f.id);
    if (it != keyframe_.landmarks.end()) {
      pts_kf.push_back(it->second);
      pts_curr.push_back(f.p_c);
      pts_2d.emplace_back(static_cast<float>(f.u_l), static_cast<float>(f.v_l));
      ++tracked_from_kf;
    }
  }

  // ------------------------------------------------------------------
  // Estimate motion via 3D-3D alignment (uses stereo depth from both frames)
  // ------------------------------------------------------------------
  if (tracked_from_kf >= params_.min_pnp_points) {
    Sophus::SE3d T_kf_curr;  // T_{kf_cam ← curr_cam}
    int inliers = 0;
    if (solveMotion3D3D(pts_kf, pts_curr, T_kf_curr, inliers)) {
      // T_{world ← curr} = T_{world ← kf} * T_{kf ← curr}
      const Sophus::SE3d T_wc_candidate = keyframe_.T_wc * T_kf_curr;

      // Motion sanity check: reject implausible jumps
      const double dt = (T_wc_candidate.translation() - T_wc_.translation()).norm();
      const Eigen::Matrix3d dR =
          T_wc_.rotationMatrix().transpose() * T_wc_candidate.rotationMatrix();
      const double angle_rad = std::acos(std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0));
      const double angle_deg = angle_rad * 180.0 / M_PI;

      if (dt < params_.max_translation_m && angle_deg < params_.max_rotation_deg) {
        T_wc_ = T_wc_candidate;
        last_inlier_count_ = inliers;
      } else {
        get_logger()->debug("StereoVO: rejected motion — dt={:.3f}m angle={:.1f}deg", dt,
                            angle_deg);
      }
    }
  }

  // ------------------------------------------------------------------
  // Keyframe management
  // ------------------------------------------------------------------
  if (shouldCreateKeyframe(tracked_from_kf, static_cast<int>(keyframe_.landmarks.size()))) {
    createKeyframe(features);
  }

  return T_wc_;
}

// ---------------------------------------------------------------------------
bool StereoVO::solvePose(const std::vector<Eigen::Vector3d>& pts_3d,
                         const std::vector<cv::Point2f>& pts_2d, Sophus::SE3d& T_ck,
                         int& inlier_count) const {
  const int N = static_cast<int>(pts_3d.size());
  if (N < params_.min_pnp_points) return false;

  // Convert 3D points to cv::Mat
  std::vector<cv::Point3f> obj_pts(N);
  for (int i = 0; i < N; ++i) {
    obj_pts[i] = cv::Point3f(static_cast<float>(pts_3d[i].x()), static_cast<float>(pts_3d[i].y()),
                             static_cast<float>(pts_3d[i].z()));
  }

  // Camera matrix (rectified intrinsics, no distortion)
  const cv::Mat K =
      (cv::Mat_<double>(3, 3) << cam_.fx, 0.0, cam_.cx, 0.0, cam_.fy, cam_.cy, 0.0, 0.0, 1.0);
  const cv::Mat dist_coeffs;  // empty = no distortion (already rectified)

  cv::Mat rvec;
  cv::Mat tvec;
  cv::Mat inliers_mat;
  const bool ok = cv::solvePnPRansac(obj_pts, pts_2d, K, dist_coeffs, rvec, tvec,
                                     false,  // useExtrinsicGuess
                                     300,    // iterationsCount
                                     static_cast<float>(params_.pnp_reproj_thresh),
                                     0.99,  // confidence
                                     inliers_mat, cv::SOLVEPNP_ITERATIVE);

  if (!ok || inliers_mat.empty()) return false;

  inlier_count = inliers_mat.rows;

  // Refine with all inliers using iterative PnP
  if (inlier_count >= params_.min_pnp_points) {
    std::vector<cv::Point3f> inlier_obj;
    std::vector<cv::Point2f> inlier_img;
    for (int i = 0; i < inlier_count; ++i) {
      const int idx = inliers_mat.at<int>(i);
      inlier_obj.push_back(obj_pts[idx]);
      inlier_img.push_back(pts_2d[idx]);
    }
    cv::solvePnP(inlier_obj, inlier_img, K, dist_coeffs, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
  }

  // Convert to Eigen
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  cv::cv2eigen(R_cv, R);
  cv::cv2eigen(tvec, t);

  T_ck = Sophus::SE3d(Eigen::Quaterniond(R), t);

  return true;
}

// ---------------------------------------------------------------------------
namespace {

// Rigid 3D-3D alignment using SVD (Horn's method, no scale)
Sophus::SE3d alignSVD(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& dst) {
  // src = points in frame A, dst = corresponding points in frame B
  // Returns T_{B <- A} such that dst[i] ≈ T * src[i]
  const int N = static_cast<int>(src.size());
  Eigen::Vector3d c_src = Eigen::Vector3d::Zero();
  Eigen::Vector3d c_dst = Eigen::Vector3d::Zero();
  for (int i = 0; i < N; ++i) {
    c_src += src[i];
    c_dst += dst[i];
  }
  c_src /= N;
  c_dst /= N;

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; ++i) {
    H += (src[i] - c_src) * (dst[i] - c_dst).transpose();
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d& V = svd.matrixV();
  const Eigen::Matrix3d& U = svd.matrixU();
  const double d = (V * U.transpose()).determinant();
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = d;  // ensure proper rotation (det=+1)
  const Eigen::Matrix3d R = V * D * U.transpose();
  const Eigen::Vector3d t = c_dst - R * c_src;
  return {Eigen::Quaterniond(R).normalized(), t};
}

}  // namespace

bool StereoVO::solveMotion3D3D(const std::vector<Eigen::Vector3d>& pts_kf,
                               const std::vector<Eigen::Vector3d>& pts_curr,
                               Sophus::SE3d& T_kf_curr, int& inlier_count) {
  const int N = static_cast<int>(pts_kf.size());
  if (N < 3) return false;

  // RANSAC
  const int max_iters = 200;
  const double thresh = 0.05;  // 5cm inlier threshold for 3D error
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, N - 1);

  int best_inliers = 0;
  Sophus::SE3d best_T;

  for (int iter = 0; iter < max_iters; ++iter) {
    // Sample 3 random correspondences
    const int i0 = dist(rng);
    int i1 = 0;
    int i2 = 0;
    do {
      i1 = dist(rng);
    } while (i1 == i0);
    do {
      i2 = dist(rng);
    } while (i2 == i0 || i2 == i1);

    const std::vector<Eigen::Vector3d> s_curr = {pts_curr[i0], pts_curr[i1], pts_curr[i2]};
    const std::vector<Eigen::Vector3d> s_kf = {pts_kf[i0], pts_kf[i1], pts_kf[i2]};

    // Fit: T_{kf <- curr} such that pts_kf[i] ≈ T * pts_curr[i]
    const Sophus::SE3d T_cand = alignSVD(s_curr, s_kf);

    // Count inliers
    int n_inliers = 0;
    for (int j = 0; j < N; ++j) {
      const Eigen::Vector3d p_kf_pred = T_cand * pts_curr[j];
      if ((p_kf_pred - pts_kf[j]).norm() < thresh) ++n_inliers;
    }

    if (n_inliers > best_inliers) {
      best_inliers = n_inliers;
      best_T = T_cand;
    }
  }

  if (best_inliers < 3) return false;

  // Refine with all inliers
  std::vector<Eigen::Vector3d> inlier_curr;
  std::vector<Eigen::Vector3d> inlier_kf;
  for (int j = 0; j < N; ++j) {
    const Eigen::Vector3d p_kf_pred = best_T * pts_curr[j];
    if ((p_kf_pred - pts_kf[j]).norm() < thresh) {
      inlier_curr.push_back(pts_curr[j]);
      inlier_kf.push_back(pts_kf[j]);
    }
  }

  T_kf_curr = alignSVD(inlier_curr, inlier_kf);
  inlier_count = static_cast<int>(inlier_curr.size());
  return true;
}

// ---------------------------------------------------------------------------
bool StereoVO::shouldCreateKeyframe(int tracked_from_kf, int total_in_kf) const {
  if (total_in_kf == 0) return true;

  // Condition 1: too few features tracked from keyframe
  const double ratio = static_cast<double>(tracked_from_kf) / static_cast<double>(total_in_kf);

  // Condition 2: sufficient parallax (median pixel displacement from KF)
  // We approximate this using the displacement between keyframe 2D positions
  // and current 2D positions for matched features.
  // (A more precise metric would use the actual angular parallax.)
  // For simplicity, just use the ratio criterion above — it works well in practice.

  return ratio < params_.kf_tracked_ratio || tracked_from_kf < params_.kf_min_tracked;
}

// ---------------------------------------------------------------------------
void StereoVO::createKeyframe(const std::vector<Feature>& features) {
  keyframe_.T_wc = T_wc_;
  keyframe_.landmarks.clear();

  for (const auto& f : features) {
    // Only store landmarks with valid depth (already filtered by tracker)
    if (f.p_c.z() > 0.2 && f.p_c.z() < 30.0) {
      keyframe_.landmarks[f.id] = f.p_c;  // 3D in keyframe camera frame
    }
  }

  has_keyframe_ = true;
}

}  // namespace ekf_vio
