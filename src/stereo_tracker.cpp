// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_tracker.hpp"

#include "ekf_vio/logging.hpp"

#include <spdlog/common.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <filesystem>
#include <format>
#include <utility>

namespace ekf_vio {

// ---------------------------------------------------------------------------
StereoTracker::StereoTracker(StereoCamera cam, const Params& p)
    : cam_(std::move(cam)),
      params_(p),
      fast_detector_(cv::FastFeatureDetector::create(p.fast_threshold)) {}

// ---------------------------------------------------------------------------
std::vector<Feature> StereoTracker::track(const cv::Mat& img_left, const cv::Mat& img_right) {
  std::vector<Feature> features;

  // -----------------------------------------------------------------------
  // 1. Track existing points from previous frame (temporal tracking)
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> curr_pts;
  std::vector<uchar> status_temporal;

  const int n_prev = static_cast<int>(prev_pts_.size());

  if (!prev_pyramid_.empty() && !prev_pts_.empty()) {
    std::vector<float> err;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    // Pass the pre-built pyramid so OpenCV skips rebuilding it for prevImg.
    // clang-format off
    cv::calcOpticalFlowPyrLK(prev_pyramid_, img_left, prev_pts_, curr_pts, status_temporal, err,
                             cv::Size(params_.lk_win_size, params_.lk_win_size),
                             params_.lk_max_level, criteria);
    // TODO:
    // if not suffienct match increase win_size
    // clang-format on

    auto n_lk_ok = std::ranges::count_if(status_temporal, [](auto s) { return s > 0; });
    get_logger()->debug("[Tracker] temporal LK: {}/{} tracked", n_lk_ok, n_prev);

    // Reject outliers via E matrix RANSAC
    if (curr_pts.size() >= 5) {
      rejectOutliers(prev_pts_, curr_pts, status_temporal);

      int n_after_ransac = 0;
      if (get_logger()->should_log(spdlog::level::debug)) {
        for (auto s : status_temporal)
          n_after_ransac += static_cast<int>(s > 0);
        get_logger()->debug("[Tracker] E-RANSAC: {}/{} inliers ({} rejected)", n_after_ransac,
                            n_lk_ok, n_lk_ok - n_after_ransac);
      }
    }
  }

  // -----------------------------------------------------------------------
  // 2. Detect new features if we lost too many
  // -----------------------------------------------------------------------
  auto valid_count = std::ranges::count_if(status_temporal, [](auto s) { return s > 0; });

  std::vector<cv::Point2f> new_pts;
  if (valid_count < params_.max_features / 2) {
    detectNew(img_left, new_pts);
    get_logger()->debug("[Tracker] detect new: {} new features (tracked={} < threshold={})",
                        new_pts.size(), valid_count, params_.max_features / 2);
  }

  // ── Debug: save temporal flow image ──────────────────────────────────────
  if (!params_.debug_save_dir.empty()) {
    const int rel = frame_count_ - params_.debug_save_start_frame;
    if (rel >= 0 && rel < params_.debug_save_count)
      saveFlowImage(img_left, prev_pts_, curr_pts, status_temporal, new_pts);
  }

  // -----------------------------------------------------------------------
  // 3. Assemble list of left keypoints (tracked + new)
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> left_pts_all;
  std::vector<int> ids_all;

  // Add successfully tracked points (carry forward persistent IDs)
  for (size_t i = 0; i < prev_pts_.size(); ++i) {
    if (i < status_temporal.size() && status_temporal[i]) {
      left_pts_all.push_back(curr_pts[i]);
      ids_all.push_back(prev_ids_[i]);
    }
  }
  // Add newly detected points
  for (auto& pt : new_pts) {
    left_pts_all.push_back(pt);
    ids_all.push_back(-1);  // new feature, will be assigned below
  }

  // -----------------------------------------------------------------------
  // 4. Stereo match: find corresponding right-image points via LK flow
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> right_pts;
  std::vector<uchar> status_stereo;
  if (params_.use_lk_stereo) {
    stereoMatchUsingOpticalFlowWithInitGuess(img_left, img_right, left_pts_all, right_pts,
                                             status_stereo);
  } else {
    stereoMatchUsingSAD(img_left, img_right, left_pts_all, right_pts, status_stereo);
  }

  // ── Debug: save stereo match image ───────────────────────────────────────
  if (!params_.debug_save_dir.empty()) {
    const int rel = frame_count_ - params_.debug_save_start_frame;
    if (rel >= 0 && rel < params_.debug_save_count)
      saveStereoImage(img_left, img_right, left_pts_all, right_pts, status_stereo);
  }

  // -----------------------------------------------------------------------
  // 5. Triangulate valid stereo matches and build output features
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> good_left_pts;
  prev_pts_.clear();

  int n_disparity_fail = 0;
  int n_depth_fail = 0;

  for (size_t i = 0; i < left_pts_all.size(); ++i) {
    if (!status_stereo[i]) continue;

    const double u_l = left_pts_all[i].x;
    const double v_l = left_pts_all[i].y;
    const double u_r = right_pts[i].x;
    const double v_r = right_pts[i].y;

    // Disparity check (horizontal stereo: left pixel is to the right of right)
    const double disparity = u_l - u_r;
    if (disparity < params_.min_disparity || disparity > params_.max_disparity) {
      ++n_disparity_fail;
      continue;
    }

    // Triangulate
    const Eigen::Vector3d p_c = triangulate(u_l, v_l, u_r);
    if (p_c.z() < 0.2 || p_c.z() > 30.0) {
      ++n_depth_fail;
      continue;
    }

    Feature feat;
    feat.id = (ids_all[i] >= 0) ? ids_all[i] : next_id_++;
    feat.u_l = u_l;
    feat.v_l = v_l;
    feat.u_r = u_r;
    feat.v_r = v_r;
    feat.p_c = p_c;
    features.push_back(feat);

    good_left_pts.push_back(left_pts_all[i]);
  }

  get_logger()->debug(
      "[Tracker] triangulate: {} ok  |  disparity_fail={}  depth_fail={}  total_out={}",
      features.size(), n_disparity_fail, n_depth_fail, features.size());

  // -----------------------------------------------------------------------
  // 6. Update state for next iteration
  // -----------------------------------------------------------------------
  // Build and cache the pyramid for img_left so the next temporal LK call
  // can use it directly without rebuilding (saves one pyramid build/frame).
  cv::buildOpticalFlowPyramid(img_left, prev_pyramid_,
                              cv::Size(params_.lk_win_size, params_.lk_win_size),
                              params_.lk_max_level);
  prev_pts_ = good_left_pts;
  prev_ids_.clear();
  for (const auto& f : features)
    prev_ids_.push_back(f.id);

  ++frame_count_;
  return features;
}

// ---------------------------------------------------------------------------
void StereoTracker::detectNew(const cv::Mat& img, std::vector<cv::Point2f>& pts) {
  // Build a mask that suppresses existing feature locations
  cv::Mat mask = cv::Mat::ones(img.size(), CV_8U) * 255;
  for (auto& pt : prev_pts_) {
    cv::circle(mask, pt, 10, 0, -1);
  }

  std::vector<cv::KeyPoint> kps;
  fast_detector_->detect(img, kps, mask);

  // Sort by response, take strongest
  std::ranges::sort(kps, [](auto& a, auto& b) { return a.response > b.response; });
  const int need = params_.max_features - static_cast<int>(prev_pts_.size());
  const int take = std::min(need, static_cast<int>(kps.size()));

  for (int i = 0; i < take; ++i) {
    pts.push_back(kps[i].pt);
  }
}

// ---------------------------------------------------------------------------
void StereoTracker::stereoMatchUsingOpticalFlowWithInitGuess(
    const cv::Mat& left, const cv::Mat& right, const std::vector<cv::Point2f>& left_pts,
    std::vector<cv::Point2f>& right_pts, std::vector<uchar>& status) const {
  if (left_pts.empty()) {
    status.clear();
    return;
  }

  // Horizontal stereo: LK flow
  right_pts = left_pts;
  std::vector<float> err;
  const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(left, right, left_pts, right_pts, status, err,
                           cv::Size(params_.lk_win_size, params_.lk_win_size), params_.lk_max_level,
                           criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  int n_lk_ok = 0;
  int n_epipolar_fail = 0;
  for (auto s : status)
    n_lk_ok += static_cast<int>(s > 0);

  // Verify epipolar constraint (same row for horizontal stereo)
  for (size_t i = 0; i < left_pts.size(); ++i) {
    if (!status[i]) continue;
    if (std::abs(left_pts[i].y - right_pts[i].y) > params_.epipolar_thresh_px) {
      status[i] = 0;
      ++n_epipolar_fail;
    }
  }

  get_logger()->debug("[Tracker] stereoMatch: lk_ok={}/{}  epipolar_fail={}  final_ok={}", n_lk_ok,
                      left_pts.size(), n_epipolar_fail, n_lk_ok - n_epipolar_fail);
}

void StereoTracker::stereoMatchUsingSAD(const cv::Mat& left, const cv::Mat& right,
                                        const std::vector<cv::Point2f>& left_pts,
                                        std::vector<cv::Point2f>& right_pts,
                                        std::vector<uchar>& status) const {
  status.assign(left_pts.size(), 0);
  right_pts.assign(left_pts.size(), cv::Point2f(0.0f, 0.0f));

  const int win_size = 11;
  const int half_win = win_size / 2;
  constexpr float kMinNCC = 0.85f;  // ZNCC acceptance threshold (illumination-invariant)

  for (size_t i = 0; i < left_pts.size(); ++i) {
    const float lx = left_pts[i].x;
    const float ly = left_pts[i].y;

    const int u_L = static_cast<int>(std::round(lx));
    const int v_L = static_cast<int>(std::round(ly));

    if (u_L < half_win || u_L >= left.cols - half_win || v_L < half_win ||
        v_L >= left.rows - half_win)
      continue;

    // 1. Left patch
    const cv::Mat patch_L = left(cv::Rect(u_L - half_win, v_L - half_win, win_size, win_size));

    // 2. 1D search strip: initial guess = same x as left (mirrors stereoMatch's USE_INITIAL_FLOW).
    //    Tight window [u_L - stereo_search_radius, u_L] rather than the full disparity range.
    const int x_start = std::max(half_win, u_L - params_.stereo_search_radius);
    const int strip_w = u_L - x_start + 1;  // number of candidate positions

    if (strip_w < 1) continue;

    // Build the right search strip: 1 row tall, strip_w + win_size - 1 wide
    const cv::Mat strip =
        right(cv::Rect(x_start - half_win, v_L - half_win, strip_w + win_size - 1, win_size));

    // cv::matchTemplate with CCOEFF_NORMED = ZNCC: illumination-invariant, score in [-1, 1]
    cv::Mat response;
    cv::matchTemplate(strip, patch_L, response, cv::TM_CCOEFF_NORMED);
    // response is 1 × strip_w

    double max_val{};
    cv::Point max_loc;
    cv::minMaxLoc(response, nullptr, &max_val, nullptr, &max_loc);

    if (static_cast<float>(max_val) < kMinNCC) continue;

    // 3. Parabolic sub-pixel refinement on the NCC response curve
    auto refined_x = static_cast<float>(x_start + max_loc.x);
    const int col = max_loc.x;
    if (col > 0 && col < response.cols - 1) {
      const float r_m1 = response.at<float>(0, col - 1);
      const float r_0 = response.at<float>(0, col);
      const float r_p1 = response.at<float>(0, col + 1);
      const float denom = 2.0f * (r_m1 - 2.0f * r_0 + r_p1);
      if (std::abs(denom) > 1e-5f) {
        refined_x += (r_m1 - r_p1) / denom;  // NCC: higher is better, same parabola formula
      }
    }

    right_pts[i] = cv::Point2f(refined_x, ly);
    status[i] = 1;
  }
}

// ---------------------------------------------------------------------------
Eigen::Vector3d StereoTracker::triangulate(double u_l, double v_l, double u_r) const {
  // Stereo triangulation (rectified horizontal stereo):
  const double disp = u_l - u_r;
  const double Z = cam_.fx * cam_.baseline / disp;
  const double X = (u_l - cam_.cx) * Z / cam_.fx;
  const double Y = (v_l - cam_.cy) * Z / cam_.fy;
  return {X, Y, Z};
}

// ---------------------------------------------------------------------------
void StereoTracker::rejectOutliers(std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& curr,
                                   std::vector<uchar>& status) const {
  std::vector<cv::Point2f> p_in;
  std::vector<cv::Point2f> c_in;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      p_in.push_back(prev[i]);
      c_in.push_back(curr[i]);
    }
  }

  // min 5 pts for finding E
  if (p_in.size() < 5) return;

  const cv::Mat K =
      (cv::Mat_<double>(3, 3) << cam_.fx, 0.0, cam_.cx, 0.0, cam_.fy, cam_.cy, 0.0, 0.0, 1.0);
  std::vector<uchar> ransac_status;
  cv::findEssentialMat(p_in, c_in, K, cv::RANSAC, 0.99, params_.ransac_thresh_px, ransac_status);

  // Map ransac_status back to full status vector
  int idx = 0;
  for (auto& s : status) {
    if (s) {
      s = ransac_status[idx++];
    }
  }
}

// only debug code from here so it doesn't count into Code Cov
// ---------------------------------------------------------------------------
// LCOV_EXCL_START
void StereoTracker::saveFlowImage(const cv::Mat& img_left, const std::vector<cv::Point2f>& prev_pts,
                                  const std::vector<cv::Point2f>& curr_pts,
                                  const std::vector<uchar>& status_temporal,
                                  const std::vector<cv::Point2f>& new_pts) const {
  std::filesystem::create_directories(params_.debug_save_dir);

  cv::Mat vis;
  cv::cvtColor(img_left, vis, cv::COLOR_GRAY2BGR);

  // Draw flow arrows: green = tracked features (prev → curr)
  for (size_t i = 0; i < prev_pts.size(); ++i) {
    if (i >= status_temporal.size() || !status_temporal[i]) continue;
    cv::arrowedLine(vis, prev_pts[i], curr_pts[i], cv::Scalar(0, 220, 0), 1, cv::LINE_AA, 0, 0.3);
    cv::circle(vis, curr_pts[i], 3, cv::Scalar(0, 220, 0), -1, cv::LINE_AA);
  }
  // Blue circles: newly detected features
  for (const auto& pt : new_pts)
    cv::circle(vis, pt, 3, cv::Scalar(255, 100, 0), -1, cv::LINE_AA);

  const auto path = std::filesystem::path(params_.debug_save_dir) /
                    std::format("flow_{:04d}.png", frame_count_ - params_.debug_save_start_frame);
  cv::imwrite(path.string(), vis);
  get_logger()->debug("[Tracker] saved flow debug image: {}", path.string());
}

// ---------------------------------------------------------------------------
void StereoTracker::saveStereoImage(const cv::Mat& img_left, const cv::Mat& img_right,
                                    const std::vector<cv::Point2f>& left_pts,
                                    const std::vector<cv::Point2f>& right_pts,
                                    const std::vector<uchar>& status_stereo) const {
  std::filesystem::create_directories(params_.debug_save_dir);

  // Side-by-side canvas
  cv::Mat vis(img_left.rows, img_left.cols + img_right.cols, CV_8UC3);
  cv::Mat left_roi = vis(cv::Rect(0, 0, img_left.cols, img_left.rows));
  cv::Mat right_roi = vis(cv::Rect(img_left.cols, 0, img_right.cols, img_right.rows));
  cv::cvtColor(img_left, left_roi, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img_right, right_roi, cv::COLOR_GRAY2BGR);

  // Divider line
  cv::line(vis, {img_left.cols, 0}, {img_left.cols, vis.rows - 1}, cv::Scalar(60, 60, 60), 1);

  for (size_t i = 0; i < left_pts.size(); ++i) {
    if (!status_stereo[i]) continue;
    const cv::Point2f r_shifted(right_pts[i].x + static_cast<float>(img_left.cols), right_pts[i].y);
    // Connecting line (dim cyan)
    cv::line(vis, left_pts[i], r_shifted, cv::Scalar(120, 120, 40), 1, cv::LINE_AA);
    // Left: green dot; right: orange dot
    cv::circle(vis, left_pts[i], 3, cv::Scalar(0, 220, 0), -1, cv::LINE_AA);
    cv::circle(vis, r_shifted, 3, cv::Scalar(0, 140, 255), -1, cv::LINE_AA);
  }

  const auto path = std::filesystem::path(params_.debug_save_dir) /
                    std::format("stereo_{:04d}.png", frame_count_ - params_.debug_save_start_frame);
  cv::imwrite(path.string(), vis);
  get_logger()->debug("[Tracker] saved stereo debug image: {}", path.string());
}
// LCOV_EXCL_STOP

}  // namespace ekf_vio
