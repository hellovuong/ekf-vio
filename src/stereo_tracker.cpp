// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_tracker.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <utility>

namespace ekf_vio {

// ---------------------------------------------------------------------------
StereoTracker::StereoTracker(StereoCamera cam, const Params& p)
    : cam_(std::move(cam)), params_(p) {}

// ---------------------------------------------------------------------------
std::vector<Feature> StereoTracker::track(const cv::Mat& img_left, const cv::Mat& img_right) {
  std::vector<Feature> features;

  // -----------------------------------------------------------------------
  // 1. Track existing points from previous frame (temporal tracking)
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> curr_pts;
  std::vector<uchar> status_temporal;

  if (!prev_left_.empty() && !prev_pts_.empty()) {
    std::vector<float> err;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_left_, img_left, prev_pts_, curr_pts, status_temporal, err,
                             cv::Size(params_.lk_win_size, params_.lk_win_size),
                             params_.lk_max_level, criteria);

    // Reject outliers via fundamental matrix RANSAC
    if (curr_pts.size() >= 8) {
      rejectOutliers(prev_pts_, curr_pts, status_temporal);
    }
  }

  // -----------------------------------------------------------------------
  // 2. Detect new features if we lost too many
  // -----------------------------------------------------------------------
  int valid_count = 0;
  for (auto s : status_temporal) {
    valid_count += static_cast<int>(s > 0);
  }

  std::vector<cv::Point2f> new_pts;
  if (valid_count < params_.max_features / 2) {
    detectNew(img_left, new_pts);
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
  //    (works for rectified stereo: flow is purely horizontal)
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> right_pts;
  std::vector<uchar> status_stereo;
  stereoMatch(img_left, img_right, left_pts_all, right_pts, status_stereo);

  // -----------------------------------------------------------------------
  // 5. Triangulate valid stereo matches and build output features
  // -----------------------------------------------------------------------
  std::vector<cv::Point2f> good_left_pts;
  prev_pts_.clear();

  for (size_t i = 0; i < left_pts_all.size(); ++i) {
    if (!status_stereo[i]) continue;

    const double u_l = left_pts_all[i].x;
    const double v_l = left_pts_all[i].y;
    const double u_r = right_pts[i].x;
    const double v_r = right_pts[i].y;

    // Disparity check (horizontal stereo: left pixel is to the right of right)
    const double disparity = u_l - u_r;
    if (disparity < params_.min_disparity || disparity > params_.max_disparity) continue;

    // Triangulate
    const Eigen::Vector3d p_c = triangulate(u_l, v_l, u_r, v_r);
    if (p_c.z() < 0.2 || p_c.z() > 30.0) continue;  // depth sanity

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

  // -----------------------------------------------------------------------
  // 6. Update state for next iteration
  // -----------------------------------------------------------------------
  img_left.copyTo(prev_left_);
  prev_pts_ = good_left_pts;
  prev_ids_.clear();
  for (const auto& f : features)
    prev_ids_.push_back(f.id);

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
  auto fast = cv::FastFeatureDetector::create(params_.fast_threshold);
  fast->detect(img, kps, mask);

  // Sort by response, take strongest
  std::sort(kps.begin(), kps.end(), [](auto& a, auto& b) { return a.response > b.response; });
  const int need = params_.max_features - static_cast<int>(prev_pts_.size());
  const int take = std::min(need, static_cast<int>(kps.size()));

  for (int i = 0; i < take; ++i) {
    pts.push_back(kps[i].pt);
  }
}

// ---------------------------------------------------------------------------
void StereoTracker::stereoMatch(const cv::Mat& left, const cv::Mat& right,
                                const std::vector<cv::Point2f>& left_pts,
                                std::vector<cv::Point2f>& right_pts,
                                std::vector<uchar>& status) const {
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
                           criteria);

  // Verify epipolar constraint (same row for horizontal stereo)
  for (size_t i = 0; i < left_pts.size(); ++i) {
    if (!status[i]) continue;
    if (std::abs(left_pts[i].y - right_pts[i].y) > 2.0) status[i] = 0;
  }
}

// ---------------------------------------------------------------------------
Eigen::Vector3d StereoTracker::triangulate(double u_l, double v_l, double u_r,
                                           double /*v_r*/) const {
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
  if (p_in.size() < 8) return;

  std::vector<uchar> ransac_status;
  cv::findFundamentalMat(p_in, c_in, ransac_status, cv::FM_RANSAC, params_.ransac_thresh_px, 0.99);

  // Map ransac_status back to full status vector
  int idx = 0;
  for (auto& s : status) {
    if (s) {
      s = ransac_status[idx++];
    }
  }
}

}  // namespace ekf_vio
