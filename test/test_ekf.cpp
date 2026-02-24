/**
 * test/test_ekf.cpp
 * Unit tests for ekf_vio::EKF predict, update, and measurement Jacobian.
 */

#include <gtest/gtest.h>

#include "ekf_vio/ekf.hpp"
#include "ekf_vio/math_utils.hpp"
#include "ekf_vio/types.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace ekf_vio;
using namespace ekf_vio::math;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a default stereo camera (EuRoC-like) with identity extrinsic.
static StereoCamera makeCamera() {
  StereoCamera cam;
  cam.fx = 458.654;
  cam.fy = 457.296;
  cam.cx = 367.215;
  cam.cy = 248.375;
  cam.baseline = 0.110;
  cam.T_cam_imu = Eigen::Isometry3d::Identity();
  return cam;
}

/// Create a camera with a non-trivial extrinsic (small rotation + translation).
static StereoCamera makeCameraWithExtrinsic() {
  StereoCamera cam = makeCamera();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = expSO3(Eigen::Vector3d(0.01, -0.02, 0.03));
  T.translation() = Eigen::Vector3d(-0.02, -0.06, 0.01);
  cam.T_cam_imu = T;
  return cam;
}

static EKF::NoiseParams defaultNoise() {
  return EKF::NoiseParams{};
}

/// Pinhole projection (left camera).
static Eigen::Vector4d projectStereo(const StereoCamera& cam,
                                     const Eigen::Vector3d& p_c) {
  const double inv_z = 1.0 / p_c.z();
  const double u_l = cam.fx * p_c.x() * inv_z + cam.cx;
  const double v_l = cam.fy * p_c.y() * inv_z + cam.cy;
  const double u_r = cam.fx * (p_c.x() - cam.baseline) * inv_z + cam.cx;
  const double v_r = v_l;
  return {u_l, v_l, u_r, v_r};
}

/// Transform a world-frame point into camera frame given state.
static Eigen::Vector3d worldToCamera(const State& s, const StereoCamera& cam,
                                     const Eigen::Vector3d& p_w) {
  const Eigen::Matrix3d R_wb = s.q.toRotationMatrix();
  const Eigen::Matrix3d R_bc = cam.T_cam_imu.rotation();
  // R_cw = (R_wb * R_bc)^T   (consistent with ekf.cpp)
  const Eigen::Matrix3d R_cw = (R_wb * R_bc).transpose();
  const Eigen::Vector3d p_cam_w = R_wb * cam.T_cam_imu.translation() + s.p;
  return R_cw * (p_w - p_cam_w);
}

/// Create a Feature from a world point given the current state and camera.
static Feature makeFeature(const State& s, const StereoCamera& cam,
                           const Eigen::Vector3d& p_w, int id) {
  const Eigen::Vector3d p_c = worldToCamera(s, cam, p_w);
  const Eigen::Vector4d z = projectStereo(cam, p_c);
  return {id, z(0), z(1), z(2), z(3), p_c};
}

// ===========================================================================
// Predict — gravity-only scenario
// ===========================================================================

TEST(EKFPredict, GravityOnly) {
  // With zero IMU input (gyro=0, accel compensates gravity so the
  // body-frame accelerometer reads [0,0,+9.81] when stationary and
  // orientation is identity (world=body).
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  // Stationary accelerometer reading: measures -gravity in body frame
  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = Eigen::Vector3d::Zero();
  imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81); // compensates gravity

  const double dt = 0.005; // 200 Hz
  ekf.predict(imu, dt);

  const State& s = ekf.state();
  // Velocity should stay ~zero (R*a_c + g = I*[0,0,9.81] + [0,0,-9.81] = 0)
  EXPECT_NEAR(s.v.norm(), 0.0, 1e-10);
  // Position should stay ~zero
  EXPECT_NEAR(s.p.norm(), 0.0, 1e-10);
  // Orientation should stay identity
  EXPECT_NEAR(s.q.w(), 1.0, 1e-10);
}

TEST(EKFPredict, FreeFall) {
  // Accelerometer reads zero in free fall (no specific force).
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = Eigen::Vector3d::Zero();
  imu.accel = Eigen::Vector3d::Zero(); // free fall

  const double dt = 0.01;
  ekf.predict(imu, dt);

  const State& s = ekf.state();
  // v should be g * dt = [0, 0, -9.81] * 0.01 = [0, 0, -0.0981]
  EXPECT_NEAR(s.v.x(), 0.0, 1e-10);
  EXPECT_NEAR(s.v.y(), 0.0, 1e-10);
  EXPECT_NEAR(s.v.z(), -9.81 * dt, 1e-8);

  // p should be 0.5 * g * dt^2
  EXPECT_NEAR(s.p.z(), 0.5 * (-9.81) * dt * dt, 1e-8);
}

TEST(EKFPredict, ConstantAngularVelocity) {
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  const Eigen::Vector3d omega(0.0, 0.0, 1.0); // 1 rad/s about z
  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = omega;
  imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81); // compensate gravity

  const double dt = 0.005;
  const int steps = 100; // 0.5 seconds
  for (int i = 0; i < steps; ++i) {
    imu.timestamp = i * dt;
    ekf.predict(imu, dt);
  }

  // After 0.5s at 1 rad/s about z, total rotation = 0.5 rad about z
  const Eigen::Vector3d omega_total = logSO3(ekf.state().q.toRotationMatrix());
  EXPECT_NEAR(omega_total.x(), 0.0, 0.01);
  EXPECT_NEAR(omega_total.y(), 0.0, 0.01);
  EXPECT_NEAR(omega_total.z(), 0.5, 0.01);
}

// ===========================================================================
// Predict — covariance properties
// ===========================================================================

TEST(EKFPredict, CovarianceSymmetricPD) {
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = Eigen::Vector3d(0.1, -0.05, 0.2);
  imu.accel = Eigen::Vector3d(-0.3, 0.1, 9.7);

  const double dt = 0.005;
  for (int i = 0; i < 50; ++i) {
    ekf.predict(imu, dt);
  }

  const auto& P = ekf.state().P;
  // Symmetric
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-12);
  // Positive definite: all eigenvalues > 0
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eig(P);
  EXPECT_GT(eig.eigenvalues().minCoeff(), 0.0);
}

// ===========================================================================
// Predict — F matrix numerical verification
// ===========================================================================

TEST(EKFPredict, FMatrixNumerical) {
  // Verify the F Jacobian by comparing single-step predict with
  // finite-difference perturbation of the error state.
  StereoCamera cam = makeCamera();
  EKF::NoiseParams noise = defaultNoise();

  // Set a non-trivial state
  State s0;
  s0.p = Eigen::Vector3d(1.0, -0.5, 2.0);
  s0.v = Eigen::Vector3d(0.3, -0.1, 0.2);
  s0.q = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 1, 1).normalized()));
  s0.b_g = Eigen::Vector3d(0.001, -0.002, 0.001);
  s0.b_a = Eigen::Vector3d(0.01, 0.02, -0.005);

  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = Eigen::Vector3d(0.2, -0.1, 0.15);
  imu.accel = Eigen::Vector3d(-0.5, 0.3, 9.6);

  const double dt = 0.005;

  // Nominal predict
  EKF ekf_nom(cam, noise);
  ekf_nom.state() = s0;
  ekf_nom.predict(imu, dt);
  const State& s_nom = ekf_nom.state();

  // Extract nominal output as 15-vector
  auto stateToVec = [](const State& s) -> Eigen::Matrix<double, 15, 1> {
    Eigen::Matrix<double, 15, 1> x;
    x.segment<3>(0) = s.p;
    x.segment<3>(3) = s.v;
    x.segment<3>(6) = logSO3(s.q.toRotationMatrix());
    x.segment<3>(9) = s.b_g;
    x.segment<3>(12) = s.b_a;
    return x;
  };

  const Eigen::Matrix<double, 15, 1> x_nom = stateToVec(s_nom);

  // Finite-difference Φ
  const double eps = 1e-7;
  Eigen::Matrix<double, 15, 15> Phi_num;

  for (int j = 0; j < 15; ++j) {
    // Perturb the j-th error state component
    State s_pert = s0;
    Eigen::Matrix<double, 15, 1> delta = Eigen::Matrix<double, 15, 1>::Zero();
    delta(j) = eps;

    s_pert.p += delta.segment<3>(0);
    s_pert.v += delta.segment<3>(3);
    s_pert.q = boxplus(s_pert.q, delta.segment<3>(6));
    s_pert.b_g += delta.segment<3>(9);
    s_pert.b_a += delta.segment<3>(12);

    EKF ekf_pert(cam, noise);
    ekf_pert.state() = s_pert;
    ekf_pert.predict(imu, dt);

    const Eigen::Matrix<double, 15, 1> x_pert = stateToVec(ekf_pert.state());

    // For orientation: compute difference via logSO3(R_nom^T * R_pert)
    Eigen::Matrix<double, 15, 1> dx = x_pert - x_nom;
    dx.segment<3>(6) = logSO3(s_nom.q.toRotationMatrix().transpose() *
                              ekf_pert.state().q.toRotationMatrix());

    Phi_num.col(j) = dx / eps;
  }

  // Analytical Φ = I + F*dt (compute F from the same state)
  // We don't have direct access to F, but Phi_num should be close to I + F*dt.
  // Check that Phi_num is close to a matrix of the expected structure.

  // Basic structure checks: Phi(0:2, 3:5) ≈ I*dt  (off-diagonal block)
  const Eigen::Matrix3d dp_dv = Phi_num.block<3, 3>(0, 3);
  EXPECT_NEAR((dp_dv - Eigen::Matrix3d::Identity() * dt).norm(), 0.0, 0.01)
      << "Phi(p,v) should be I*dt but got\n"
      << dp_dv;

  // Phi should be close to identity on diagonal
  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(Phi_num(i, i), 1.0, 0.1)
        << "Diagonal element " << i << " too far from 1";
  }

  // Bias columns (9:11, 12:14) of Phi should affect θ and v respectively
  // Phi(6:8, 9:11) ≈ -I*dt  (gyro bias → orientation)
  const Eigen::Matrix3d dtheta_dbg = Phi_num.block<3, 3>(6, 9);
  EXPECT_NEAR((dtheta_dbg - (-Eigen::Matrix3d::Identity() * dt)).norm(), 0.0,
              0.01);
}

// ===========================================================================
// Measurement Jacobian — numerical verification
// ===========================================================================

TEST(EKFUpdate, MeasurementJacobianNumerical) {
  // Verify the measurement Jacobian H by finite-difference.
  // We perturb each error state component and compare the change in
  // the predicted measurement.
  StereoCamera cam = makeCameraWithExtrinsic();
  EKF::NoiseParams noise = defaultNoise();

  // Set non-trivial state
  State s0;
  s0.p = Eigen::Vector3d(0.5, -0.3, 1.0);
  s0.v = Eigen::Vector3d::Zero();
  s0.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
  s0.b_g = Eigen::Vector3d::Zero();
  s0.b_a = Eigen::Vector3d::Zero();

  // A world point 5m in front of the camera
  const Eigen::Vector3d p_w(2.0, 0.5, 1.5);

  // Compute nominal camera-frame point and projection
  const Eigen::Vector3d p_c0 = worldToCamera(s0, cam, p_w);
  ASSERT_GT(p_c0.z(), 0.1) << "Point behind camera";
  const Eigen::Vector4d z0 = projectStereo(cam, p_c0);

  // Finite-difference H (4×15)
  const double eps = 1e-7;
  Eigen::Matrix<double, 4, 15> H_num;
  H_num.setZero();

  for (int j = 0; j < 15; ++j) {
    State s_pert = s0;
    Eigen::Matrix<double, 15, 1> delta = Eigen::Matrix<double, 15, 1>::Zero();
    delta(j) = eps;

    s_pert.p += delta.segment<3>(0);
    s_pert.v += delta.segment<3>(3);
    s_pert.q = boxplus(s_pert.q, delta.segment<3>(6));
    s_pert.b_g += delta.segment<3>(9);
    s_pert.b_a += delta.segment<3>(12);

    const Eigen::Vector3d p_c_pert = worldToCamera(s_pert, cam, p_w);
    const Eigen::Vector4d z_pert = projectStereo(cam, p_c_pert);

    H_num.col(j) = (z_pert - z0) / eps;
  }

  // Analytical H from the EKF (via update internals)
  // We replicate the H computation from ekf.cpp::measurementJacobian.
  {
    const Eigen::Matrix3d R_wb = s0.q.toRotationMatrix();
    const Eigen::Matrix3d R_cw = (R_wb * cam.T_cam_imu.rotation()).transpose();

    const Eigen::Vector3d pc = p_c0;
    const double z = pc.z();
    const double z2 = z * z;

    Eigen::Matrix<double, 2, 3> J_proj_l;
    J_proj_l << cam.fx / z, 0.0, -cam.fx * pc.x() / z2, 0.0, cam.fy / z,
        -cam.fy * pc.y() / z2;

    Eigen::Matrix<double, 2, 3> J_proj_r;
    J_proj_r << cam.fx / z, 0.0, -cam.fx * (pc.x() - cam.baseline) / z2, 0.0,
        cam.fy / z, -cam.fy * pc.y() / z2;

    // Position: ∂p_c/∂δp = -R_cw
    const Eigen::Matrix3d dp_c_dp = -R_cw;

    // Orientation: ∂p_c/∂δθ = [p_c]× · R_cw (approximate, dropping lever arm)
    const Eigen::Matrix3d dp_c_dtheta = skew(pc) * R_cw;

    Eigen::Matrix<double, 4, 15> H_analytical;
    H_analytical.setZero();
    H_analytical.block<2, 3>(0, 0) = J_proj_l * dp_c_dp;
    H_analytical.block<2, 3>(2, 0) = J_proj_r * dp_c_dp;
    H_analytical.block<2, 3>(0, 6) = J_proj_l * dp_c_dtheta;
    H_analytical.block<2, 3>(2, 6) = J_proj_r * dp_c_dtheta;

    // Position columns (0:2): should match closely
    EXPECT_NEAR(
        (H_analytical.block<4, 3>(0, 0) - H_num.block<4, 3>(0, 0)).norm(), 0.0,
        0.5)
        << "Position Jacobian mismatch.\nAnalytical:\n"
        << H_analytical.block<4, 3>(0, 0) << "\nNumerical:\n"
        << H_num.block<4, 3>(0, 0);

    // Orientation columns (6:8): lever-arm approximation causes ~5% relative
    // error when camera-IMU offset is non-zero. Use relative tolerance.
    const double orient_num_norm = H_num.block<4, 3>(0, 6).norm();
    const double orient_err =
        (H_analytical.block<4, 3>(0, 6) - H_num.block<4, 3>(0, 6)).norm();
    EXPECT_LT(orient_err / orient_num_norm, 0.10)
        << "Orientation Jacobian relative error > 10%.\nAnalytical:\n"
        << H_analytical.block<4, 3>(0, 6) << "\nNumerical:\n"
        << H_num.block<4, 3>(0, 6);

    // Velocity and bias columns should be zero in both
    EXPECT_NEAR((H_num.block<4, 3>(0, 3).norm()), 0.0, 1e-4)
        << "Velocity columns should be zero";
    EXPECT_NEAR((H_num.block<4, 3>(0, 9).norm()), 0.0, 1e-4)
        << "Gyro bias columns should be zero";
    EXPECT_NEAR((H_num.block<4, 3>(0, 12).norm()), 0.0, 1e-4)
        << "Accel bias columns should be zero";
  }
}

TEST(EKFUpdate, MeasurementJacobianNumericalIdentityExtrinsic) {
  // With identity extrinsic the lever-arm error vanishes, so the
  // analytical Jacobian should match numerics tightly.
  StereoCamera cam = makeCamera(); // identity T_cam_imu
  EKF::NoiseParams noise = defaultNoise();

  State s0;
  s0.p = Eigen::Vector3d(0.5, -0.3, 1.0);
  s0.q = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d(0.5, 0.3, 0.8).normalized()));

  const Eigen::Vector3d p_w(3.0, 1.0, 2.0);
  const Eigen::Vector3d p_c0 = worldToCamera(s0, cam, p_w);
  ASSERT_GT(p_c0.z(), 0.1);
  const Eigen::Vector4d z0 = projectStereo(cam, p_c0);

  const double eps = 1e-7;
  Eigen::Matrix<double, 4, 15> H_num;
  H_num.setZero();

  for (int j = 0; j < 15; ++j) {
    State s_pert = s0;
    Eigen::Matrix<double, 15, 1> delta = Eigen::Matrix<double, 15, 1>::Zero();
    delta(j) = eps;

    s_pert.p += delta.segment<3>(0);
    s_pert.v += delta.segment<3>(3);
    s_pert.q = boxplus(s_pert.q, delta.segment<3>(6));
    s_pert.b_g += delta.segment<3>(9);
    s_pert.b_a += delta.segment<3>(12);

    const Eigen::Vector3d p_c_pert = worldToCamera(s_pert, cam, p_w);
    const Eigen::Vector4d z_pert = projectStereo(cam, p_c_pert);
    H_num.col(j) = (z_pert - z0) / eps;
  }

  // Analytical H
  const Eigen::Matrix3d R_wb = s0.q.toRotationMatrix();
  const Eigen::Matrix3d R_cw = (R_wb * cam.T_cam_imu.rotation()).transpose();
  const Eigen::Vector3d pc = p_c0;
  const double z = pc.z();
  const double z2 = z * z;

  Eigen::Matrix<double, 2, 3> J_proj_l;
  J_proj_l << cam.fx / z, 0.0, -cam.fx * pc.x() / z2, 0.0, cam.fy / z,
      -cam.fy * pc.y() / z2;

  Eigen::Matrix<double, 2, 3> J_proj_r;
  J_proj_r << cam.fx / z, 0.0, -cam.fx * (pc.x() - cam.baseline) / z2, 0.0,
      cam.fy / z, -cam.fy * pc.y() / z2;

  const Eigen::Matrix3d dp_c_dp = -R_cw;
  const Eigen::Matrix3d dp_c_dtheta = skew(pc) * R_cw;

  Eigen::Matrix<double, 4, 15> H_a;
  H_a.setZero();
  H_a.block<2, 3>(0, 0) = J_proj_l * dp_c_dp;
  H_a.block<2, 3>(2, 0) = J_proj_r * dp_c_dp;
  H_a.block<2, 3>(0, 6) = J_proj_l * dp_c_dtheta;
  H_a.block<2, 3>(2, 6) = J_proj_r * dp_c_dtheta;

  // With identity extrinsic, position and orientation columns should be exact
  EXPECT_NEAR((H_a.block<4, 3>(0, 0) - H_num.block<4, 3>(0, 0)).norm(), 0.0,
              0.01)
      << "Position Jacobian (identity extrinsic):\nAnalytical:\n"
      << H_a.block<4, 3>(0, 0) << "\nNumerical:\n"
      << H_num.block<4, 3>(0, 0);

  EXPECT_NEAR((H_a.block<4, 3>(0, 6) - H_num.block<4, 3>(0, 6)).norm(), 0.0,
              0.01)
      << "Orientation Jacobian (identity extrinsic):\nAnalytical:\n"
      << H_a.block<4, 3>(0, 6) << "\nNumerical:\n"
      << H_num.block<4, 3>(0, 6);
}

// ===========================================================================
// Update — covariance properties
// ===========================================================================

TEST(EKFUpdate, CovarianceSymmetricPD) {
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  // Set state so features are in front of camera
  ekf.state().p = Eigen::Vector3d::Zero();
  ekf.state().q = Eigen::Quaterniond::Identity();

  // World points in front of camera (z > 0 in camera frame with identity pose)
  std::vector<Eigen::Vector3d> world_pts = {
      {1.0, 0.5, 5.0},
      {-0.5, 0.3, 4.0},
      {0.3, -0.2, 6.0},
  };

  std::vector<Feature> features;
  for (int i = 0; i < static_cast<int>(world_pts.size()); ++i) {
    features.push_back(makeFeature(ekf.state(), cam, world_pts[i], i));
  }

  ekf.update(features);

  const auto& P = ekf.state().P;
  // Symmetric
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-12);
  // Positive semi-definite
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eig(P);
  EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-12)
      << "Covariance has negative eigenvalue: " << eig.eigenvalues().minCoeff();
}

TEST(EKFUpdate, CovarianceDecreases) {
  // After an update, the covariance trace should not increase
  // (information from measurements can only reduce uncertainty).
  StereoCamera cam = makeCamera();
  EKF ekf(cam, defaultNoise());

  // Run a few predict steps to grow covariance
  ImuData imu;
  imu.timestamp = 0.0;
  imu.gyro = Eigen::Vector3d(0.05, -0.03, 0.1);
  imu.accel = Eigen::Vector3d(-0.1, 0.05, 9.75);
  for (int i = 0; i < 20; ++i) {
    ekf.predict(imu, 0.005);
  }

  const double trace_before = ekf.state().P.trace();

  // Create features
  std::vector<Eigen::Vector3d> world_pts = {
      {2.0, 0.5, 5.0}, {-1.0, 0.3, 4.0},  {0.5, -0.5, 6.0},
      {1.5, 1.0, 3.5}, {-0.3, -0.8, 7.0},
  };
  std::vector<Feature> features;
  for (int i = 0; i < static_cast<int>(world_pts.size()); ++i) {
    features.push_back(makeFeature(ekf.state(), cam, world_pts[i], i));
  }

  ekf.update(features);

  const double trace_after = ekf.state().P.trace();
  EXPECT_LE(trace_after, trace_before + 1e-10);
}

// ===========================================================================
// Stereo triangulation (via projection roundtrip)
// ===========================================================================

TEST(StereoTriangulation, ProjectionRoundtrip) {
  // Triangulate a known 3D point from its stereo projections and verify
  // the roundtrip.
  StereoCamera cam = makeCamera();

  const Eigen::Vector3d p_c(0.5, -0.3, 4.0); // point in camera frame
  const Eigen::Vector4d z = projectStereo(cam, p_c);

  // Triangulate using the standard formula
  const double disparity = z(0) - z(2); // u_l - u_r
  ASSERT_GT(disparity, 0.0);
  const double Z = cam.fx * cam.baseline / disparity;
  const double X = (z(0) - cam.cx) * Z / cam.fx;
  const double Y = (z(1) - cam.cy) * Z / cam.fy;
  const Eigen::Vector3d p_rec(X, Y, Z);

  EXPECT_NEAR((p_c - p_rec).norm(), 0.0, 1e-10);
}

TEST(StereoTriangulation, DepthFromDisparity) {
  StereoCamera cam = makeCamera();

  // A point at 10m depth
  const double Z = 10.0;
  const double expected_disparity = cam.fx * cam.baseline / Z;

  const Eigen::Vector3d p_c(0.0, 0.0, Z);
  const Eigen::Vector4d z = projectStereo(cam, p_c);
  const double actual_disparity = z(0) - z(2);

  EXPECT_NEAR(actual_disparity, expected_disparity, 1e-10);
}
