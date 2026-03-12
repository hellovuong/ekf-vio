// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_tracker.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
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
  std::ranges::sort(z);
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

// ---------------------------------------------------------------------------
// ZNCC-specific tests
// ---------------------------------------------------------------------------

// ZNCC uses zero-mean NCC which subtracts the per-patch mean before
// correlation, making it invariant to uniform brightness scaling between
// the left and right cameras.  Plain SAD would fail this test.
TEST(StereoTrackerZncc, IlluminationInvariantMatch) {
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

  // Scale right image by 1.5x to simulate different camera gain.
  // ZNCC subtracts the patch mean so this should not affect matching quality.
  cv::Mat right_gained;
  right.convertTo(right_gained, CV_8U, 1.5, 0);

  const auto features = tracker.track(left, right_gained);

  ASSERT_GT(features.size(), 40u);
  const double expected_z = cam.fx * cam.baseline / static_cast<double>(disparity_px);
  EXPECT_NEAR(medianDepth(features), expected_z, 0.5);
}

// A random-noise right image has no structural correlation with the left.
// The NCC acceptance threshold (≥ 0.85) should reject all candidates,
// yielding zero (or near-zero) matched features.
TEST(StereoTrackerZncc, RejectsRandomRightImage) {
  const auto cam = makeCamera();
  ekf_vio::StereoTracker::Params p;
  p.max_features = 300;
  p.fast_threshold = 15;
  p.min_disparity = 1.0;
  p.max_disparity = 200.0;

  ekf_vio::StereoTracker tracker(cam, p);

  cv::Mat left;
  cv::Mat dummy;
  makeShiftedStereoPair(640, 480, 20, left, dummy);  // only need left

  cv::Mat right_noise(480, 640, CV_8U);
  cv::RNG rng(42);
  rng.fill(right_noise, cv::RNG::UNIFORM, 0, 255);

  const auto features = tracker.track(left, right_noise);

  EXPECT_LT(features.size(), 5u);
}

// When the true disparity exceeds stereo_search_radius the search strip
// [u_L - search_radius, u_L] never covers the true right-image position,
// so no candidates pass the NCC threshold.
TEST(StereoTrackerZncc, LargeDisparityBeyondSearchRadiusRejected) {
  const auto cam = makeCamera();
  ekf_vio::StereoTracker::Params p;
  p.max_features = 300;
  p.fast_threshold = 15;
  p.min_disparity = 1.0;
  p.max_disparity = 200.0;
  p.stereo_search_radius = 20;  // tight window

  ekf_vio::StereoTracker tracker(cam, p);

  cv::Mat left;
  cv::Mat right;
  // True disparity 60 px >> search_radius 20 px → strip misses the match.
  makeShiftedStereoPair(640, 480, 60, left, right);

  const auto features = tracker.track(left, right);

  EXPECT_LT(features.size(), 5u);
}

// ---------------------------------------------------------------------------

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
