// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_tracker.hpp"

#include <opencv2/core.hpp>

#include <gtest/gtest.h>
#include <unordered_set>

namespace {

ekf_vio::StereoCamera makeCamera() {
  ekf_vio::StereoCamera cam;
  cam.fx = 436.0;
  cam.fy = 436.0;
  cam.cx = 320.0;
  cam.cy = 240.0;
  cam.baseline = 0.11;
  cam.T_cam_imu = Sophus::SE3d();
  return cam;
}

void makeShiftedStereoPair(int w, int h, int disparity_px, cv::Mat& left, cv::Mat& right) {
  left = cv::Mat(h, w, CV_8U);
  cv::RNG rng(7);
  rng.fill(left, cv::RNG::UNIFORM, 0, 255);

  right = cv::Mat::zeros(h, w, CV_8U);
  // right(u, v) = left(u + d, v)  => correspondence: u_r = u_l - d
  left.colRange(disparity_px, w).copyTo(right.colRange(0, w - disparity_px));
}

double medianDepth(const std::vector<ekf_vio::Feature>& features) {
  std::vector<double> z;
  z.reserve(features.size());
  for (const auto& f : features)
    z.push_back(f.p_c.z());
  std::sort(z.begin(), z.end());
  return z[z.size() / 2];
}

}  // namespace

TEST(StereoTrackerTest, TriangulatesExpectedDepthFromKnownDisparity) {
  const auto cam = makeCamera();
  ekf_vio::StereoTracker::Params p;
  p.max_features = 300;
  p.fast_threshold = 15;
  p.min_disparity = 2.0;
  p.max_disparity = 120.0;

  ekf_vio::StereoTracker tracker(cam, p);

  cv::Mat left;
  cv::Mat right;
  const int disparity_px = 28;
  makeShiftedStereoPair(640, 480, disparity_px, left, right);

  const auto features = tracker.track(left, right);
  ASSERT_GT(features.size(), 80u);

  const double expected_z = cam.fx * cam.baseline / static_cast<double>(disparity_px);
  const double med_z = medianDepth(features);
  EXPECT_NEAR(med_z, expected_z, 0.25);
}

TEST(StereoTrackerTest, KeepsFeatureIdsAcrossFrames) {
  const auto cam = makeCamera();
  ekf_vio::StereoTracker::Params p;
  p.max_features = 300;
  p.fast_threshold = 15;
  p.min_disparity = 2.0;
  p.max_disparity = 120.0;

  ekf_vio::StereoTracker tracker(cam, p);

  cv::Mat left;
  cv::Mat right;
  makeShiftedStereoPair(640, 480, 24, left, right);

  const auto f1 = tracker.track(left, right);
  const auto f2 = tracker.track(left, right);  // same pair -> IDs should persist

  ASSERT_GT(f1.size(), 80u);
  ASSERT_GT(f2.size(), 80u);

  std::unordered_set<int> ids1;
  for (const auto& f : f1)
    ids1.insert(f.id);

  int overlap = 0;
  for (const auto& f : f2) {
    if (ids1.contains(f.id)) ++overlap;
  }

  const int denom = std::min(static_cast<int>(f1.size()), static_cast<int>(f2.size()));
  EXPECT_GT(overlap, static_cast<int>(0.8 * denom));
}
