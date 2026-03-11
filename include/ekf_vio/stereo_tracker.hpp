// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "ekf_vio/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <map>
#include <string>
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
    double min_disparity = 1.0;  // pixels (reject degenerate stereo)
    double max_disparity = 200.0;
    int stereo_search_radius = 50;    // ZNCC 1-D search half-width (px) around left position
    double ransac_thresh_px = 1.5;    // for outlier rejection via E-matrix
    double epipolar_thresh_px = 3.0;  // max vertical disparity for rectified stereo match
    double min_track_quality = 0.001;

    // Debug image saving (empty = disabled)
    // Saves debug_save_count frames starting at debug_save_start_frame:
    //   <dir>/flow_NNNN.png   — temporal LK flow (arrows prev→curr, blue = new detections)
    //   <dir>/stereo_NNNN.png — stereo match (left|right side-by-side with match lines)
    std::string debug_save_dir;      // empty = disabled
    int debug_save_start_frame = 0;  // first frame index to save
    int debug_save_count = 5;        // how many consecutive frames to save
  };

  explicit StereoTracker(StereoCamera cam, const Params& p);

  // -----------------------------------------------------------------------
  // Feed a new stereo pair.  Returns tracked + triangulated features.
  // Call this at camera rate.
  // -----------------------------------------------------------------------
  std::vector<Feature> track(const cv::Mat& img_left, const cv::Mat& img_right);

 private:
  // Detect new FAST corners in regions sparse of existing tracks
  void detectNew(const cv::Mat& img, std::vector<cv::Point2f>& pts);

  // Stereo match via horizontal LK optical flow (rectified stereo assumed).
  // Fast pyramid-based; uses left pixel as initial guess for right.
  // Followed by an epipolar row-consistency check (|Δv| ≤ epipolar_thresh_px).
  void stereoMatchUsingOpticalFlowWithInitGuess(const cv::Mat& left, const cv::Mat& right,
                                                const std::vector<cv::Point2f>& left_pts,
                                                std::vector<cv::Point2f>& right_pts,
                                                std::vector<uchar>& status) const;

  // Stereo match via Zero-mean Normalized Cross-Correlation (ZNCC) patch matching.
  //
  // Exploits rectified stereo geometry: the right-image correspondent of a left
  // point (u_L, v_L) lies on the same row (v_R = v_L) at a disparity
  // d = u_L - u_R ∈ [min_disparity, max_disparity].  The search is therefore
  // strictly 1-D.
  //
  // Algorithm per feature:
  //   1. Extract an 11×11 patch centred on (u_L, v_L) from the left image.
  //   2. Build a 1-row search strip in the right image spanning
  //      [u_L - stereo_search_radius, u_L] on the same row. The right feature is assumed to be near
  //      the same x-position as the left (small-to-moderate disparity).  This
  //      keeps the strip tight (~50 px vs ~200 px), reducing both runtime and
  //      false-match probability.
  //   3. Run cv::matchTemplate(TM_CCOEFF_NORMED) across the strip — equivalent
  //      to ZNCC.  Score ∈ [-1, 1]; subtracting per-patch means makes it
  //      invariant to per-camera gain/exposure differences (unlike plain SAD).
  //   4. Accept the best candidate only if its NCC score ≥ 0.85.
  //   5. Apply parabolic sub-pixel refinement on the NCC response curve around
  //      the peak, achieving sub-pixel disparity accuracy.
  //
  // Compared to stereoMatch() (LK optical flow):
  //   + Illumination-invariant: handles left/right exposure mismatch robustly
  //     (key advantage on challenging sequences such as V1_02_medium).
  //   − Slower: O(N × max_disparity × win²) vs LK's pyramid convergence.
  void stereoMatchUsingSAD(const cv::Mat& left, const cv::Mat& right,
                           const std::vector<cv::Point2f>& left_pts,
                           std::vector<cv::Point2f>& right_pts, std::vector<uchar>& status) const;

  // Triangulate from stereo (assuming rectified cameras)
  [[nodiscard]] Eigen::Vector3d triangulate(double u_l, double v_l, double u_r) const;

  // Outlier rejection with fundamental matrix RANSAC
  void rejectOutliers(std::vector<cv::Point2f>& prev, std::vector<cv::Point2f>& curr,
                      std::vector<uchar>& status) const;

  StereoCamera cam_;
  Params params_;

  // Cached FAST detector — avoids create() allocation on every detectNew() call
  cv::Ptr<cv::FastFeatureDetector> fast_detector_;

  // Previous frame data
  // Pyramid is pre-built and stored so the next temporal LK call reuses it
  // rather than rebuilding it from the raw image (saves one pyramid build/frame).
  std::vector<cv::Mat> prev_pyramid_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<int> prev_ids_;            // persistent feature IDs (parallel to prev_pts_)
  std::map<int, cv::Point2f> id_to_pt_;  // feature id → left pixel
  int next_id_ = 0;
  int frame_count_ = 0;  // incremented each track() call; used for debug save indexing

  void saveFlowImage(const cv::Mat& img_left, const std::vector<cv::Point2f>& prev_pts,
                     const std::vector<cv::Point2f>& curr_pts,
                     const std::vector<uchar>& status_temporal,
                     const std::vector<cv::Point2f>& new_pts) const;

  void saveStereoImage(const cv::Mat& img_left, const cv::Mat& img_right,
                       const std::vector<cv::Point2f>& left_pts,
                       const std::vector<cv::Point2f>& right_pts,
                       const std::vector<uchar>& status_stereo) const;
};

}  // namespace ekf_vio
