// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "ekf_vio/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <map>
#include <vector>

namespace ekf_vio {

// ===========================================================================
//  StereoTracker
//
//  Tracks features across:
//    (a) Stereo pairs   : left ↔ right  (via block matching / LK flow)
//    (b) Temporal pairs : prev ↔ curr   (via Lucas-Kanade optical flow)
//
//  Output: vector<Feature> with stereo pixel coords + triangulated 3-D point
// ===========================================================================
class StereoTracker {
 public:
  struct Params {
    int max_features = 300;
    int fast_threshold = 20;
    int lk_win_size = 21;
    int lk_max_level = 3;        // pyramid levels
    double min_disparity = 1.0;  // pixels  (reject degenerate stereo)
    double max_disparity = 200.0;
    double ransac_thresh_px = 1.5;  // for outlier rejection via F-matrix
    double min_track_quality = 0.001;
  };

  explicit StereoTracker(const StereoCamera& cam, const Params& p);

  // -----------------------------------------------------------------------
  // Feed a new stereo pair.  Returns tracked + triangulated features.
  // Call this at camera rate.
  // -----------------------------------------------------------------------
  std::vector<Feature> track(const cv::Mat& img_left, const cv::Mat& img_right);

 private:
  // Detect new FAST corners in regions sparse of existing tracks
  void detectNew(const cv::Mat& img, std::vector<cv::Point2f>& pts);

  // Stereo match via horizontal LK flow (rectified stereo assumed)
  void stereoMatch(const cv::Mat& left, const cv::Mat& right,
                   const std::vector<cv::Point2f>& left_pts, std::vector<cv::Point2f>& right_pts,
                   std::vector<uchar>& status);

  // Triangulate from stereo (assuming rectified cameras)
  Eigen::Vector3d triangulate(double u_l, double v_l, double u_r, double v_r) const;

  // Outlier rejection with fundamental matrix RANSAC
  void rejectOutliers(std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& curr,
                      std::vector<uchar>& status);

  StereoCamera cam_;
  Params params_;

  // Previous frame data
  cv::Mat prev_left_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<int> prev_ids_;            // persistent feature IDs (parallel to prev_pts_)
  std::map<int, cv::Point2f> id_to_pt_;  // feature id → left pixel
  int next_id_ = 0;
};

}  // namespace ekf_vio
