// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_rectifier.hpp"

#include "ekf_vio/logging.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ekf_vio {

void StereoRectifier::init(const CameraConfig& ccfg) {
  const auto& c0 = ccfg.cam0;
  const auto& c1 = ccfg.cam1;

  const cv::Mat K0 =
      (cv::Mat_<double>(3, 3) << c0.fx, 0.0, c0.cx, 0.0, c0.fy, c0.cy, 0.0, 0.0, 1.0);
  const cv::Mat D0 = (cv::Mat_<double>(4, 1) << c0.distortion[0], c0.distortion[1],
                      c0.distortion[2], c0.distortion[3]);

  const cv::Mat K1 =
      (cv::Mat_<double>(3, 3) << c1.fx, 0.0, c1.cx, 0.0, c1.fy, c1.cy, 0.0, 0.0, 1.0);
  const cv::Mat D1 = (cv::Mat_<double>(4, 1) << c1.distortion[0], c1.distortion[1],
                      c1.distortion[2], c1.distortion[3]);

  // T_{cam1<-cam0} = inv(T_BS1) * T_BS0
  Eigen::Matrix4d T_rel = ccfg.T_BS1.inverse() * ccfg.T_BS0;
  cv::Mat R_cv(3, 3, CV_64F);
  cv::Mat T_cv(3, 1, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c)
      R_cv.at<double>(r, c) = T_rel(r, c);
    T_cv.at<double>(r) = T_rel(r, 3);
  }

  const cv::Size imgSize(ccfg.image_width, ccfg.image_height);
  cv::Mat R1;
  cv::Mat R2;
  cv::Mat P1;
  cv::Mat P2;
  cv::Mat Q;
  cv::stereoRectify(K0, D0, K1, D1, imgSize, R_cv, T_cv, R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, 0.0, imgSize);

  cv::initUndistortRectifyMap(K0, D0, R1, P1, imgSize, CV_32FC1, map1_left_, map2_left_);
  cv::initUndistortRectifyMap(K1, D1, R2, P2, imgSize, CV_32FC1, map1_right_, map2_right_);

  // Store R1 for T_cam_imu correction
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R_rect_(r, c) = R1.at<double>(r, c);
    }
  }

  const double f_rect = P1.at<double>(0, 0);
  fx_ = f_rect;
  fy_ = f_rect;
  cx_ = P1.at<double>(0, 2);
  cy_ = P1.at<double>(1, 2);
  baseline_ = std::abs(P2.at<double>(0, 3)) / f_rect;

  get_logger()->info("Rectify: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f} baseline={:.4f}m", fx_, fy_,
                     cx_, cy_, baseline_);
}

void StereoRectifier::rectify(const cv::Mat& raw_left, const cv::Mat& raw_right, cv::Mat& rect_left,
                              cv::Mat& rect_right) const {
  cv::remap(raw_left, rect_left, map1_left_, map2_left_, cv::INTER_LINEAR);
  cv::remap(raw_right, rect_right, map1_right_, map2_right_, cv::INTER_LINEAR);
}

// ── Factory helpers ──────────────────────────────────────────

StereoCamera makeStereoCamera(const StereoRectifier& rect, const CameraConfig& ccfg) {
  StereoCamera cam;
  cam.fx = rect.fx();
  cam.fy = rect.fy();
  cam.cx = rect.cx();
  cam.cy = rect.cy();
  cam.baseline = rect.baseline();

  // T_{rect_cam←imu} = R1 * T_BS0^{-1}
  Eigen::Matrix4d T_rect = Eigen::Matrix4d::Identity();
  T_rect.block<3, 3>(0, 0) = rect.R_rect();
  cam.T_cam_imu = Sophus::SE3d(T_rect * ccfg.T_BS0.inverse());

  return cam;
}

StereoCamera makeStereoCamera(const StereoRectifier& rect) {
  StereoCamera cam;
  cam.fx = rect.fx();
  cam.fy = rect.fy();
  cam.cx = rect.cx();
  cam.cy = rect.cy();
  cam.baseline = rect.baseline();
  cam.T_cam_imu = Sophus::SE3d();
  return cam;
}

}  // namespace ekf_vio
