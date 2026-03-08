/**
 * ekf_vio/stereo_rectifier.hpp — Stereo rectification from calibration config.
 *
 * Computes rectification maps from CameraConfig and provides:
 *   - Rectified intrinsics (fx, fy, cx, cy, baseline)
 *   - Image rectification via remap
 *   - Factory helpers to build StereoCamera structs
 */
#pragma once

#include "ekf_vio/config.hpp"
#include "ekf_vio/types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace ekf_vio {

class StereoRectifier {
public:
  /// Compute rectification maps from camera config.
  void init(const CameraConfig &ccfg);

  /// Rectify a raw stereo pair.
  void rectify(const cv::Mat &raw_left, const cv::Mat &raw_right,
               cv::Mat &rect_left, cv::Mat &rect_right) const;

  /// Rectified intrinsics (valid after init()).
  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double baseline() const { return baseline_; }

  /// Rectification rotation applied to left camera (R1).
  const Eigen::Matrix3d &R_rect() const { return R_rect_; }

private:
  cv::Mat map1_left_, map2_left_;
  cv::Mat map1_right_, map2_right_;
  double fx_{}, fy_{}, cx_{}, cy_{}, baseline_{};
  Eigen::Matrix3d R_rect_ = Eigen::Matrix3d::Identity();
};

// ── Factory helpers ──────────────────────────────────────────

/// Build StereoCamera with T_cam_imu computed from rectification + T_BS0.
/// Use this for VIO (needs cam↔IMU extrinsic).
StereoCamera makeStereoCamera(const StereoRectifier &rect,
                               const CameraConfig &ccfg);

/// Build StereoCamera with identity T_cam_imu (VO-only mode, no IMU).
StereoCamera makeStereoCamera(const StereoRectifier &rect);

} // namespace ekf_vio
