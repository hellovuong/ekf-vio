// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/ekf.hpp"

#include "ekf_vio/math_utils.hpp"
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
  // Non-trivial cam-imu extrinsic: 90° rotation about Z
  cam.T_cam_imu =
      Sophus::SE3d(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
                   Eigen::Vector3d(0.05, -0.02, 0.01));
  return cam;
}

ekf_vio::EKF::NoiseParams defaultNoise() {
  ekf_vio::EKF::NoiseParams n;
  n.sigma_gyro = 1.7e-4;
  n.sigma_accel = 2.0e-3;
  n.sigma_gyro_bias = 1.9e-5;
  n.sigma_accel_bias = 3.0e-5;
  n.sigma_pixel = 1.5;
  return n;
}

// Build synthetic features (random landmarks in front of the camera)
std::vector<ekf_vio::Feature> makeSyntheticFeatures(const ekf_vio::StereoCamera& cam, int count,
                                                    int base_id = 0, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dz(2.0, 6.0);
  std::uniform_real_distribution<double> dx(-1.5, 1.5);
  std::uniform_real_distribution<double> dy(-1.0, 1.0);

  std::vector<ekf_vio::Feature> features;
  for (int i = 0; i < count; ++i) {
    // Random point in camera frame
    const Eigen::Vector3d p_c(dx(rng), dy(rng), dz(rng));
    const double inv_z = 1.0 / p_c.z();
    const double u_l = (cam.fx * p_c.x() * inv_z) + cam.cx;
    const double v_l = (cam.fy * p_c.y() * inv_z) + cam.cy;
    const double u_r = (cam.fx * (p_c.x() - cam.baseline) * inv_z) + cam.cx;
    const double v_r = v_l;

    if (u_l < 0 || u_l > 640 || v_l < 0 || v_l > 480) continue;
    if ((u_l - u_r) < 1.0) continue;

    ekf_vio::Feature f;
    f.id = base_id + i;
    f.u_l = u_l;
    f.v_l = v_l;
    f.u_r = u_r;
    f.v_r = v_r;
    f.p_c = p_c;
    features.push_back(f);
  }
  return features;
}

}  // namespace

// ==========================================================================
// Test: stationary IMU — accelerometer reads gravity, position stays put
// ==========================================================================
TEST(EKFTest, StationaryIMUKeepsPosition) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  // Start at origin, identity orientation, zero velocity
  ekf.state().T_wb = Sophus::SE3d();
  ekf.state().v = Eigen::Vector3d::Zero();
  ekf.state().b_g = Eigen::Vector3d::Zero();
  ekf.state().b_a = Eigen::Vector3d::Zero();

  // Stationary: accel measures -g in body frame (body Z = world Z for identity q)
  ekf_vio::ImuData imu;
  imu.gyro = Eigen::Vector3d::Zero();
  imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81);  // opposing gravity

  const double dt = 0.005;          // 200 Hz
  for (int i = 0; i < 2000; ++i) {  // 10 seconds
    imu.timestamp = i * dt;
    ekf.predict(imu, dt);
  }

  // Position and velocity should remain near zero
  EXPECT_NEAR(ekf.state().T_wb.translation().norm(), 0.0, 0.01);
  EXPECT_NEAR(ekf.state().v.norm(), 0.0, 0.01);
}

// ==========================================================================
// Test: free-fall — zero accel → constant velocity + gravity acceleration
// ==========================================================================
TEST(EKFTest, FreeFallUnderGravity) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  ekf.state().T_wb = Sophus::SE3d();
  ekf.state().v = Eigen::Vector3d(1.0, 0.0, 0.0);  // 1 m/s along X
  ekf.state().b_g = Eigen::Vector3d::Zero();
  ekf.state().b_a = Eigen::Vector3d::Zero();

  // Free fall: accel reads zero (only gravity acts)
  ekf_vio::ImuData imu;
  imu.gyro = Eigen::Vector3d::Zero();
  imu.accel = Eigen::Vector3d::Zero();

  const double dt = 0.005;
  const double T = 1.0;  // 1 second
  for (int i = 0; i < static_cast<int>(T / dt); ++i) {
    imu.timestamp = i * dt;
    ekf.predict(imu, dt);
  }

  // After 1s: p_x = v0*t = 1.0, p_z = -0.5*g*t^2 = -4.905
  // v_x = 1.0, v_z = -g*t = -9.81
  EXPECT_NEAR(ekf.state().T_wb.translation().x(), 1.0, 0.02);
  EXPECT_NEAR(ekf.state().T_wb.translation().y(), 0.0, 0.02);
  EXPECT_NEAR(ekf.state().T_wb.translation().z(), -0.5 * 9.81, 0.05);
  EXPECT_NEAR(ekf.state().v.x(), 1.0, 0.02);
  EXPECT_NEAR(ekf.state().v.z(), -9.81, 0.05);
}

// ==========================================================================
// Test: constant angular velocity → correct rotation
// ==========================================================================
TEST(EKFTest, ConstantAngularVelocity) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  ekf.state().T_wb = Sophus::SE3d();
  ekf.state().v = Eigen::Vector3d::Zero();
  ekf.state().b_g = Eigen::Vector3d::Zero();
  ekf.state().b_a = Eigen::Vector3d::Zero();

  // Rotate at 10 deg/s about Z for 1 second
  const double omega_z = 10.0 * M_PI / 180.0;  // rad/s
  ekf_vio::ImuData imu;
  imu.gyro = Eigen::Vector3d(0.0, 0.0, omega_z);
  imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81);  // stationary

  const double dt = 0.005;
  const double T = 1.0;
  for (int i = 0; i < static_cast<int>(T / dt); ++i) {
    imu.timestamp = i * dt;
    ekf.predict(imu, dt);
  }

  // Expected: 10° yaw
  const double expected_yaw = omega_z * T;
  const Eigen::Matrix3d R_expected =
      Eigen::AngleAxisd(expected_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d R_actual = ekf.state().T_wb.rotationMatrix();
  const Eigen::Matrix3d dR = R_expected.transpose() * R_actual;
  const double angle_err = std::acos(std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0));
  EXPECT_NEAR(angle_err * 180.0 / M_PI, 0.0, 0.1);  // < 0.1°
}

// ==========================================================================
// Test: covariance grows during prediction
// ==========================================================================
TEST(EKFTest, CovarianceGrowsDuringPrediction) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  ekf.state().P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-4;

  const double trace_before = ekf.state().P.trace();

  ekf_vio::ImuData imu;
  imu.gyro = Eigen::Vector3d::Zero();
  imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81);

  const double dt = 0.005;
  for (int i = 0; i < 100; ++i) {
    imu.timestamp = i * dt;
    ekf.predict(imu, dt);
  }

  const double trace_after = ekf.state().P.trace();
  EXPECT_GT(trace_after, trace_before);
}

// ==========================================================================
// Test: camToWorld / worldToCam round-trip consistency
// ==========================================================================
TEST(EKFTest, CamWorldRoundTrip) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  // Set non-trivial state
  ekf.state().T_wb =
      Sophus::SE3d(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix(),
                   Eigen::Vector3d(1.0, 2.0, 3.0));

  // A point in camera frame
  const Eigen::Vector3d p_c(0.5, -0.3, 4.0);

  // cam → world → cam should give back the same point
  // We need access to camToWorld and worldToCam, which are private.
  // Instead, test indirectly via feature update: initialise landmark,
  // then predict its camera-frame position.
  // For this, we test that a feature placed at a world point projects
  // to the expected pixel location.

  // Actually, let's use the public API: create a feature, update (initialises landmark),
  // then update again (should have near-zero residual since nothing moved)

  auto features = makeSyntheticFeatures(cam, 50);
  ASSERT_GT(features.size(), 20u);

  const double trace_before = ekf.state().P.trace();
  ekf.update(features);  // initialises landmarks (no measurement update on first sight)

  // Second call: now features are tracked → measurement update
  ekf.update(features);

  // State shouldn't change much (residuals should be near zero)
  EXPECT_NEAR(ekf.state().T_wb.translation().x(), 1.0, 0.05);
  EXPECT_NEAR(ekf.state().T_wb.translation().y(), 2.0, 0.05);
  EXPECT_NEAR(ekf.state().T_wb.translation().z(), 3.0, 0.05);

  // Covariance should shrink after update
  const double trace_after = ekf.state().P.trace();
  EXPECT_LT(trace_after, trace_before);
}

// ==========================================================================
// Test: visual update with consistent synthetic features shrinks covariance
// ==========================================================================
TEST(EKFTest, VisualUpdateShrinksCovarianceWithConsistentFeatures) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  ekf.state().T_wb = Sophus::SE3d();
  ekf.state().v = Eigen::Vector3d::Zero();

  // Inflate covariance
  ekf.state().P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-2;

  auto features = makeSyntheticFeatures(cam, 80);
  ASSERT_GT(features.size(), 30u);

  ekf.update(features);  // initialise landmarks
  const double trace_init = ekf.state().P.trace();

  ekf.update(features);  // measurement update
  const double trace_updated = ekf.state().P.trace();

  EXPECT_LT(trace_updated, trace_init);
}

// ==========================================================================
// Test: updateFromPose corrects a position error
// ==========================================================================
TEST(EKFTest, PoseUpdateCorrectsPositionError) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  // True pose
  const Sophus::SE3d T_true(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 2.0, 3.0));

  // EKF starts with a 0.5m position error
  ekf.state().T_wb = Sophus::SE3d(Eigen::Matrix3d::Identity(),
                                  T_true.translation() + Eigen::Vector3d(0.5, -0.3, 0.2));
  ekf.state().v = Eigen::Vector3d::Zero();
  ekf.state().P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-1;

  // Give it a perfect pose measurement
  ekf.updateFromPose(T_true, 0.01, 0.01);

  // Position should move towards the true value
  EXPECT_NEAR(ekf.state().T_wb.translation().x(), T_true.translation().x(), 0.1);
  EXPECT_NEAR(ekf.state().T_wb.translation().y(), T_true.translation().y(), 0.1);
  EXPECT_NEAR(ekf.state().T_wb.translation().z(), T_true.translation().z(), 0.1);
}

// ==========================================================================
// Test: updateFromPose corrects an orientation error
// ==========================================================================
TEST(EKFTest, PoseUpdateCorrectsOrientationError) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());

  const Sophus::SE3d T_true;

  // 5° yaw error
  ekf.state().T_wb = Sophus::SE3d(
      Eigen::AngleAxisd(5.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
      Eigen::Vector3d::Zero());
  ekf.state().v = Eigen::Vector3d::Zero();
  ekf.state().P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-1;

  ekf.updateFromPose(T_true, 0.01, 0.01);

  // Orientation error should reduce
  const Eigen::Matrix3d dR =
      T_true.rotationMatrix().transpose() * ekf.state().T_wb.rotationMatrix();
  const double angle_err = std::acos(std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0));
  EXPECT_NEAR(angle_err * 180.0 / M_PI, 0.0, 1.0);  // < 1°
}

// ==========================================================================
// Test: measurement Jacobian numerical check (finite differences)
// ==========================================================================
TEST(EKFTest, MeasurementJacobianNumericalCheck) {
  using namespace ekf_vio;
  using namespace ekf_vio::math;

  const auto cam = makeCamera();
  const Eigen::Matrix3d R_ci = cam.T_cam_imu.rotationMatrix();
  const Eigen::Vector3d t_ci = cam.T_cam_imu.translation();

  // State
  const Eigen::Vector3d p(1.0, 0.5, 0.3);
  const Eigen::Quaterniond q(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
  const Eigen::Matrix3d R_wb = q.toRotationMatrix();
  const Eigen::Matrix3d R_cw = R_ci * R_wb.transpose();

  // A world landmark
  const Eigen::Vector3d p_w(3.0, 1.0, 2.0);

  // Camera-frame point
  const Eigen::Vector3d p_imu = R_wb.transpose() * (p_w - p);
  const Eigen::Vector3d p_c = R_ci * p_imu + t_ci;
  ASSERT_GT(p_c.z(), 0.1);

  // Analytic Jacobian for left projection w.r.t. position
  const double z_c = p_c.z();
  const double z_c2 = z_c * z_c;
  Eigen::Matrix<double, 2, 3> J_l;
  J_l << cam.fx / z_c, 0.0, -cam.fx * p_c.x() / z_c2, 0.0, cam.fy / z_c, -cam.fy * p_c.y() / z_c2;
  const Eigen::Matrix3d dp_c_dp = -R_cw;
  const Eigen::Matrix<double, 2, 3> H_p_analytic = J_l * dp_c_dp;

  // Numerical Jacobian: perturb position
  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 3> H_p_numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d p_plus = p;
    p_plus(i) += eps;
    const Eigen::Vector3d p_imu_plus = R_wb.transpose() * (p_w - p_plus);
    const Eigen::Vector3d p_c_plus = R_ci * p_imu_plus + t_ci;
    const double u_plus = (cam.fx * p_c_plus.x() / p_c_plus.z()) + cam.cx;
    const double v_plus = (cam.fy * p_c_plus.y() / p_c_plus.z()) + cam.cy;

    Eigen::Vector3d p_minus = p;
    p_minus(i) -= eps;
    const Eigen::Vector3d p_imu_minus = R_wb.transpose() * (p_w - p_minus);
    const Eigen::Vector3d p_c_minus = R_ci * p_imu_minus + t_ci;
    const double u_minus = (cam.fx * p_c_minus.x() / p_c_minus.z()) + cam.cx;
    const double v_minus = (cam.fy * p_c_minus.y() / p_c_minus.z()) + cam.cy;

    H_p_numeric(0, i) = (u_plus - u_minus) / (2.0 * eps);
    H_p_numeric(1, i) = (v_plus - v_minus) / (2.0 * eps);
  }

  EXPECT_NEAR((H_p_analytic - H_p_numeric).norm(), 0.0, 1e-4);

  // Analytic Jacobian for left projection w.r.t. orientation
  const Eigen::Matrix3d dp_c_dtheta = R_ci * skew(p_imu);
  const Eigen::Matrix<double, 2, 3> H_theta_analytic = J_l * dp_c_dtheta;

  // Numerical: perturb orientation (body-frame)
  Eigen::Matrix<double, 2, 3> H_theta_numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
    dtheta(i) = eps;

    // R_plus = R_wb * exp([dtheta]x)
    const Eigen::Matrix3d R_plus = R_wb * expSO3(dtheta);
    const Eigen::Vector3d p_imu_plus = R_plus.transpose() * (p_w - p);
    const Eigen::Vector3d p_c_plus = R_ci * p_imu_plus + t_ci;
    const double u_plus = (cam.fx * p_c_plus.x() / p_c_plus.z()) + cam.cx;
    const double v_plus = (cam.fy * p_c_plus.y() / p_c_plus.z()) + cam.cy;

    const Eigen::Matrix3d R_minus = R_wb * expSO3(-dtheta);
    const Eigen::Vector3d p_imu_minus = R_minus.transpose() * (p_w - p);
    const Eigen::Vector3d p_c_minus = R_ci * p_imu_minus + t_ci;
    const double u_minus = (cam.fx * p_c_minus.x() / p_c_minus.z()) + cam.cx;
    const double v_minus = (cam.fy * p_c_minus.y() / p_c_minus.z()) + cam.cy;

    H_theta_numeric(0, i) = (u_plus - u_minus) / (2.0 * eps);
    H_theta_numeric(1, i) = (v_plus - v_minus) / (2.0 * eps);
  }

  EXPECT_NEAR((H_theta_analytic - H_theta_numeric).norm(), 0.0, 1e-3);
}

// ==========================================================================
// Test: math_utils — expSO3 / logSO3 round-trip
// ==========================================================================
TEST(EKFTest, ExpLogSO3RoundTrip) {
  using namespace ekf_vio::math;

  // Several rotation vectors
  const std::vector<Eigen::Vector3d> omegas = {
      {0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.0, -0.2, 0.0}, {0.1, 0.2, -0.3}, {1.0, 0.5, -0.7},
  };

  for (const auto& w : omegas) {
    const Eigen::Matrix3d R = expSO3(w);
    const Eigen::Vector3d w_back = logSO3(R);
    EXPECT_NEAR((w - w_back).norm(), 0.0, 1e-10) << "Failed for omega = " << w.transpose();
  }
}

// ==========================================================================
// Test: math_utils — boxplus applies body-frame rotation
// ==========================================================================
TEST(EKFTest, BoxplusAppliesBodyFrameRotation) {
  using namespace ekf_vio::math;

  const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d dtheta(0.0, 0.0, 0.1);  // small Z rotation

  const Eigen::Quaterniond q_new = boxplus(q, dtheta);
  const Eigen::Matrix3d R_expected = Eigen::Matrix3d::Identity() * expSO3(dtheta);
  const Eigen::Matrix3d dR = R_expected.transpose() * q_new.toRotationMatrix();
  const double angle = std::acos(std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0));
  EXPECT_NEAR(angle, 0.0, 1e-10);
}

// ==========================================================================
// Test: empty features don't crash update()
// ==========================================================================
TEST(EKFTest, EmptyFeaturesNoCrash) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());
  ekf.update({});  // should not crash
  SUCCEED();
}

// ==========================================================================
// Test: P stays symmetric and positive semi-definite
// ==========================================================================
TEST(EKFTest, CovarianceStaysSymmetric) {
  const auto cam = makeCamera();
  ekf_vio::EKF ekf(cam, defaultNoise());
  ekf.state().P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-3;

  ekf_vio::ImuData imu;
  imu.gyro = Eigen::Vector3d(0.01, -0.02, 0.03);
  imu.accel = Eigen::Vector3d(0.1, -0.2, 9.81);

  for (int i = 0; i < 200; ++i) {
    imu.timestamp = i * 0.005;
    ekf.predict(imu, 0.005);
  }

  const auto& P = ekf.state().P;
  // Symmetry
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-12);
  // All eigenvalues >= 0
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eig(P);
  EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-12);
}
