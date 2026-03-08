// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/stereo_vo.hpp"

#include "ekf_vio/types.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
ekf_vio::StereoCamera makeCamera() {
  ekf_vio::StereoCamera cam;
  cam.fx = 436.0;
  cam.fy = 436.0;
  cam.cx = 320.0;
  cam.cy = 240.0;
  cam.baseline = 0.11;
  cam.T_cam_imu = Eigen::Isometry3d::Identity();
  return cam;
}

// Project a 3-D point in the left-camera frame to stereo pixel pair
void project(const ekf_vio::StereoCamera& cam, const Eigen::Vector3d& p, double& u_l, double& v_l,
             double& u_r, double& v_r) {
  const double inv_z = 1.0 / p.z();
  u_l = cam.fx * p.x() * inv_z + cam.cx;
  v_l = cam.fy * p.y() * inv_z + cam.cy;
  u_r = cam.fx * (p.x() - cam.baseline) * inv_z + cam.cx;
  v_r = v_l;
}

// Generate a grid of 3-D landmarks at various depths
std::vector<Eigen::Vector3d> makeLandmarks(int n, double z_near, double z_far, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dz(z_near, z_far);
  std::uniform_real_distribution<double> dx(-1.5, 1.5);
  std::uniform_real_distribution<double> dy(-1.0, 1.0);
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(n);
  for (int i = 0; i < n; ++i)
    pts.emplace_back(dx(rng), dy(rng), dz(rng));
  return pts;
}

// Build Features from 3-D points in camera frame (with id starting at base_id)
std::vector<ekf_vio::Feature> buildFeatures(const ekf_vio::StereoCamera& cam,
                                            const std::vector<Eigen::Vector3d>& pts_cam,
                                            int base_id = 0) {
  std::vector<ekf_vio::Feature> features;
  for (int i = 0; i < static_cast<int>(pts_cam.size()); ++i) {
    const auto& pc = pts_cam[i];
    if (pc.z() < 0.2 || pc.z() > 30.0) continue;
    ekf_vio::Feature f;
    f.id = base_id + i;
    project(cam, pc, f.u_l, f.v_l, f.u_r, f.v_r);
    // Skip points that project outside image
    if (f.u_l < 0 || f.u_l > 640 || f.v_l < 0 || f.v_l > 480) continue;
    f.p_c = pc;
    features.push_back(f);
  }
  return features;
}

// Transform a set of 3-D points by an Isometry
std::vector<Eigen::Vector3d> transform(const Eigen::Isometry3d& T,
                                       const std::vector<Eigen::Vector3d>& pts) {
  std::vector<Eigen::Vector3d> out;
  out.reserve(pts.size());
  for (const auto& p : pts)
    out.push_back(T * p);
  return out;
}

double translationError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
  return (a.translation() - b.translation()).norm();
}

double rotationErrorDeg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
  const Eigen::Matrix3d dR = a.rotation().transpose() * b.rotation();
  const double cos_angle = std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cos_angle) * 180.0 / M_PI;
}

}  // namespace

// ==========================================================================
// Test: first call to process() creates a keyframe
// ==========================================================================
TEST(StereoVOTest, FirstFrameCreatesKeyframe) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO vo(cam);

  auto lm = makeLandmarks(100, 1.5, 5.0);
  auto features = buildFeatures(cam, lm);
  ASSERT_GT(features.size(), 20u);

  vo.process(features);

  EXPECT_GT(vo.numKeyframeLandmarks(), 0);
  EXPECT_EQ(vo.numInliers(), 0);  // no motion estimated on first frame
}

// ==========================================================================
// Test: stationary camera -> pose stays at identity
// ==========================================================================
TEST(StereoVOTest, StationaryCameraKeepsIdentityPose) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  ekf_vio::StereoVO vo(cam, p);

  auto lm = makeLandmarks(150, 1.5, 5.0);
  auto features = buildFeatures(cam, lm);

  // Feed the same features for several frames
  for (int i = 0; i < 10; ++i)
    vo.process(features);

  EXPECT_NEAR(translationError(vo.pose(), Eigen::Isometry3d::Identity()), 0.0, 0.01);
  EXPECT_NEAR(rotationErrorDeg(vo.pose(), Eigen::Isometry3d::Identity()), 0.0, 0.5);
}

// ==========================================================================
// Test: 3D-3D alignment recovers known pure translation
// ==========================================================================
TEST(StereoVOTest, RecoversKnownTranslation) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  p.kf_tracked_ratio = 0.0;  // never auto-create keyframes
  p.kf_min_tracked = 0;
  ekf_vio::StereoVO vo(cam, p);

  // World landmarks
  auto lm_world = makeLandmarks(200, 2.0, 6.0);

  // Frame 0: camera at identity
  Eigen::Isometry3d T_wc0 = Eigen::Isometry3d::Identity();
  // Points in camera frame = world points (identity transform)
  auto feat0 = buildFeatures(cam, lm_world);
  ASSERT_GT(feat0.size(), 50u);

  vo.process(feat0);  // creates keyframe

  // Frame 1: camera moved 0.3m along X
  Eigen::Isometry3d T_wc1 = Eigen::Isometry3d::Identity();
  T_wc1.translation() = Eigen::Vector3d(0.3, 0.0, 0.0);
  // Points in camera frame: p_c = T_cw * p_w = T_wc^{-1} * p_w
  auto lm_cam1 = transform(T_wc1.inverse(), lm_world);
  auto feat1 = buildFeatures(cam, lm_cam1);
  // Keep same IDs as feat0
  for (auto& f : feat1) {
    // The id in buildFeatures is base_id + i, so IDs match if we use same base
    // This works because both calls use base_id=0 and same landmark order
  }
  ASSERT_GT(feat1.size(), 50u);

  vo.process(feat1);

  EXPECT_NEAR(vo.pose().translation().x(), 0.3, 0.05);
  EXPECT_NEAR(vo.pose().translation().y(), 0.0, 0.05);
  EXPECT_NEAR(vo.pose().translation().z(), 0.0, 0.05);
}

// ==========================================================================
// Test: 3D-3D alignment recovers known rotation + translation
// ==========================================================================
TEST(StereoVOTest, RecoversKnownRotationAndTranslation) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  p.kf_tracked_ratio = 0.0;
  p.kf_min_tracked = 0;
  ekf_vio::StereoVO vo(cam, p);

  auto lm_world = makeLandmarks(200, 2.0, 6.0);
  auto feat0 = buildFeatures(cam, lm_world);
  ASSERT_GT(feat0.size(), 50u);
  vo.process(feat0);

  // Frame 1: translate + 5° yaw
  Eigen::Isometry3d T_wc1 = Eigen::Isometry3d::Identity();
  const double yaw = 5.0 * M_PI / 180.0;
  T_wc1.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix();
  T_wc1.translation() = Eigen::Vector3d(0.1, -0.05, 0.2);

  auto lm_cam1 = transform(T_wc1.inverse(), lm_world);
  auto feat1 = buildFeatures(cam, lm_cam1);
  ASSERT_GT(feat1.size(), 50u);

  vo.process(feat1);

  EXPECT_NEAR(translationError(vo.pose(), T_wc1), 0.0, 0.08);
  EXPECT_NEAR(rotationErrorDeg(vo.pose(), T_wc1), 0.0, 1.5);
}

// ==========================================================================
// Test: implausible motion is rejected (motion sanity check)
// ==========================================================================
TEST(StereoVOTest, RejectsImplausibleMotion) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  p.max_translation_m = 1.0;
  p.max_rotation_deg = 20.0;
  p.kf_tracked_ratio = 0.0;
  p.kf_min_tracked = 0;
  ekf_vio::StereoVO vo(cam, p);

  auto lm_world = makeLandmarks(200, 2.0, 6.0);
  auto feat0 = buildFeatures(cam, lm_world);
  vo.process(feat0);

  // Frame 1: huge 10m jump — should be rejected
  Eigen::Isometry3d T_wc1 = Eigen::Isometry3d::Identity();
  T_wc1.translation() = Eigen::Vector3d(10.0, 0.0, 0.0);
  auto lm_cam1 = transform(T_wc1.inverse(), lm_world);
  auto feat1 = buildFeatures(cam, lm_cam1);
  vo.process(feat1);

  // Pose should still be at identity (rejected the big jump)
  EXPECT_NEAR(translationError(vo.pose(), Eigen::Isometry3d::Identity()), 0.0, 0.01);
}

// ==========================================================================
// Test: keyframe is created when tracking ratio drops
// ==========================================================================
TEST(StereoVOTest, CreatesNewKeyframeWhenTrackingDrops) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  p.kf_tracked_ratio = 0.50;
  p.kf_min_tracked = 20;
  ekf_vio::StereoVO vo(cam, p);

  // Frame 0: 100 landmarks
  auto lm0 = makeLandmarks(100, 2.0, 5.0, 42);
  auto feat0 = buildFeatures(cam, lm0);
  vo.process(feat0);
  const int kf_lm_after_f0 = vo.numKeyframeLandmarks();
  EXPECT_GT(kf_lm_after_f0, 50);

  // Frame 1: same features (still tracking well) -> no new KF
  vo.process(feat0);
  EXPECT_EQ(vo.numKeyframeLandmarks(), kf_lm_after_f0);

  // Frame 2: completely NEW set of features (different IDs) -> triggers new KF
  auto lm_new = makeLandmarks(60, 2.0, 5.0, 999);
  auto feat_new = buildFeatures(cam, lm_new, 10000);  // different base_id
  ASSERT_GT(feat_new.size(), 20u);
  vo.process(feat_new);

  // After new KF, feeding feat_new again should yield tracked inliers
  vo.process(feat_new);
  EXPECT_GT(vo.numInliers(), 0);
}

// ==========================================================================
// Test: multi-step trajectory accumulates pose correctly
// ==========================================================================
TEST(StereoVOTest, MultiStepTrajectory) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO::Params p;
  p.min_pnp_points = 6;
  p.kf_tracked_ratio = 0.0;
  p.kf_min_tracked = 0;
  p.max_translation_m = 2.0;
  p.max_rotation_deg = 30.0;
  ekf_vio::StereoVO vo(cam, p);

  auto lm_world = makeLandmarks(200, 2.0, 8.0);

  // Frame 0
  auto feat0 = buildFeatures(cam, lm_world);
  vo.process(feat0);

  // Move forward in 5 steps of 0.2m along Z
  Eigen::Isometry3d T_wc = Eigen::Isometry3d::Identity();
  for (int step = 1; step <= 5; ++step) {
    T_wc.translation() = Eigen::Vector3d(0.0, 0.0, 0.2 * step);
    auto lm_cam = transform(T_wc.inverse(), lm_world);
    auto feat = buildFeatures(cam, lm_cam);

    // Create a new keyframe at each step to avoid drift through a single KF
    // We do this by giving a small number of features with old IDs
    // and many new features to trigger KF creation
    if (step > 1) {
      // Manually trigger a new keyframe by feeding all-new IDs first
      auto feat_new = buildFeatures(cam, lm_cam, step * 10000);
      vo.process(feat_new);
    }

    vo.process(feat);
  }

  // Final position should be approximately (0, 0, 1.0)
  EXPECT_NEAR(vo.pose().translation().z(), 1.0, 0.2);
  EXPECT_NEAR(vo.pose().translation().x(), 0.0, 0.1);
  EXPECT_NEAR(vo.pose().translation().y(), 0.0, 0.1);
}

// ==========================================================================
// Test: empty features don't crash and preserve pose
// ==========================================================================
TEST(StereoVOTest, EmptyFeaturesPreservePose) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO vo(cam);

  auto lm = makeLandmarks(100, 2.0, 5.0);
  auto feat = buildFeatures(cam, lm);
  vo.process(feat);

  const auto pose_before = vo.pose();
  vo.process({});  // empty
  EXPECT_NEAR(translationError(vo.pose(), pose_before), 0.0, 1e-12);
}

// ==========================================================================
// Test: setInitialPose works
// ==========================================================================
TEST(StereoVOTest, SetInitialPose) {
  const auto cam = makeCamera();
  ekf_vio::StereoVO vo(cam);

  Eigen::Isometry3d T_init = Eigen::Isometry3d::Identity();
  T_init.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  vo.setInitialPose(T_init);

  EXPECT_NEAR(translationError(vo.pose(), T_init), 0.0, 1e-12);
}
