/**
 * test/test_math_utils.cpp
 * Unit tests for ekf_vio::math utilities (SO(3) operations).
 */

#include <gtest/gtest.h>

#include "ekf_vio/math_utils.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace ekf_vio::math;

constexpr double kTol = 1e-10;
constexpr double kTolLoose = 1e-6;

// ===========================================================================
// skew()
// ===========================================================================

TEST(Skew, Antisymmetric) {
  const Eigen::Vector3d v(1.0, -2.0, 3.0);
  const Eigen::Matrix3d S = skew(v);
  EXPECT_NEAR((S + S.transpose()).norm(), 0.0, kTol);
}

TEST(Skew, CrossProductEquivalence) {
  const Eigen::Vector3d a(0.3, -1.2, 0.7);
  const Eigen::Vector3d b(2.1, 0.5, -0.9);
  const Eigen::Vector3d cross = a.cross(b);
  const Eigen::Vector3d skew_prod = skew(a) * b;
  EXPECT_NEAR((cross - skew_prod).norm(), 0.0, kTol);
}

TEST(Skew, ZeroVector) {
  const Eigen::Matrix3d S = skew(Eigen::Vector3d::Zero());
  EXPECT_NEAR(S.norm(), 0.0, kTol);
}

// ===========================================================================
// expSO3()
// ===========================================================================

TEST(ExpSO3, IdentityAtZero) {
  const Eigen::Matrix3d R = expSO3(Eigen::Vector3d::Zero());
  EXPECT_NEAR((R - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
}

TEST(ExpSO3, KnownRotation90Z) {
  // 90° rotation about z-axis: ω = [0, 0, π/2]
  const Eigen::Vector3d omega(0.0, 0.0, M_PI / 2.0);
  const Eigen::Matrix3d R = expSO3(omega);

  Eigen::Matrix3d R_expected;
  R_expected << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  EXPECT_NEAR((R - R_expected).norm(), 0.0, kTolLoose);
}

TEST(ExpSO3, KnownRotation180X) {
  // 180° rotation about x-axis
  const Eigen::Vector3d omega(M_PI, 0.0, 0.0);
  const Eigen::Matrix3d R = expSO3(omega);

  Eigen::Matrix3d R_expected;
  R_expected << 1, 0, 0, 0, -1, 0, 0, 0, -1;
  EXPECT_NEAR((R - R_expected).norm(), 0.0, kTolLoose);
}

TEST(ExpSO3, ProperRotation) {
  // Check det(R) = 1 and R^T R = I for arbitrary rotation
  const Eigen::Vector3d omega(0.5, -0.3, 1.2);
  const Eigen::Matrix3d R = expSO3(omega);
  EXPECT_NEAR(R.determinant(), 1.0, kTol);
  EXPECT_NEAR((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 0.0,
              kTol);
}

TEST(ExpSO3, SmallAngle) {
  // Very small angle should match first-order: R ≈ I + [ω]×
  const Eigen::Vector3d omega(1e-11, -2e-11, 3e-11);
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Matrix3d R_approx = Eigen::Matrix3d::Identity() + skew(omega);
  EXPECT_NEAR((R - R_approx).norm(), 0.0, kTol);
}

TEST(ExpSO3, ConsistentWithEigen) {
  // Compare with Eigen's AngleAxis
  const Eigen::Vector3d omega(0.8, -0.4, 0.6);
  const double theta = omega.norm();
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Matrix3d R_eigen =
      Eigen::AngleAxisd(theta, omega.normalized()).toRotationMatrix();
  EXPECT_NEAR((R - R_eigen).norm(), 0.0, kTol);
}

// ===========================================================================
// logSO3()
// ===========================================================================

TEST(LogSO3, IdentityGivesZero) {
  const Eigen::Vector3d omega = logSO3(Eigen::Matrix3d::Identity());
  EXPECT_NEAR(omega.norm(), 0.0, kTol);
}

TEST(LogSO3, RoundtripSmallAngle) {
  const Eigen::Vector3d omega(0.01, -0.02, 0.03);
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Vector3d omega_rec = logSO3(R);
  EXPECT_NEAR((omega - omega_rec).norm(), 0.0, kTolLoose);
}

TEST(LogSO3, RoundtripLargeAngle) {
  const Eigen::Vector3d omega(1.5, -0.8, 0.3);
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Vector3d omega_rec = logSO3(R);
  EXPECT_NEAR((omega - omega_rec).norm(), 0.0, kTolLoose);
}

TEST(LogSO3, Roundtrip90Degrees) {
  const Eigen::Vector3d omega(0.0, M_PI / 2.0, 0.0);
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Vector3d omega_rec = logSO3(R);
  EXPECT_NEAR((omega - omega_rec).norm(), 0.0, kTolLoose);
}

TEST(LogSO3, PreservesAngle) {
  const Eigen::Vector3d omega(0.7, -1.1, 0.4);
  const double theta = omega.norm();
  const Eigen::Matrix3d R = expSO3(omega);
  const Eigen::Vector3d omega_rec = logSO3(R);
  EXPECT_NEAR(omega_rec.norm(), theta, kTolLoose);
}

// ===========================================================================
// boxplus()
// ===========================================================================

TEST(Boxplus, ZeroPerturbation) {
  // boxplus(q, 0) = q
  const Eigen::Quaterniond q =
      Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond q2 = boxplus(q, Eigen::Vector3d::Zero());
  // Quaternions may differ by sign
  EXPECT_NEAR(std::abs(q.dot(q2)), 1.0, kTol);
}

TEST(Boxplus, IdentityQuaternion) {
  // boxplus(identity, dθ) = exp(dθ)
  const Eigen::Quaterniond q_id = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d dtheta(0.1, -0.2, 0.3);
  const Eigen::Quaterniond q2 = boxplus(q_id, dtheta);
  const Eigen::Matrix3d R2 = q2.toRotationMatrix();
  const Eigen::Matrix3d R_exp = expSO3(dtheta);
  EXPECT_NEAR((R2 - R_exp).norm(), 0.0, kTol);
}

TEST(Boxplus, InversePerturbation) {
  // boxplus(q, dθ) then boxplus(result, -dθ) ≈ q  (for small dθ)
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 1, 0).normalized()));
  const Eigen::Vector3d dtheta(0.01, -0.005, 0.008);
  const Eigen::Quaterniond q2 = boxplus(q, dtheta);
  const Eigen::Quaterniond q3 = boxplus(q2, -dtheta);
  EXPECT_NEAR(std::abs(q.dot(q3)), 1.0, 1e-4);
}

// ===========================================================================
// leftJacobianSO3()
// ===========================================================================

TEST(LeftJacobianSO3, IdentityAtZero) {
  const Eigen::Matrix3d J = leftJacobianSO3(Eigen::Vector3d::Zero());
  EXPECT_NEAR((J - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
}

TEST(LeftJacobianSO3, NumericalComparison) {
  // Left Jacobian: exp(ω + δω) ≈ exp(J_l · δω) · exp(ω)
  // So J_l · δω ≈ logSO3( exp(ω + δω) · exp(ω)^T )
  const Eigen::Vector3d omega(0.5, -0.3, 0.8);
  const Eigen::Matrix3d J = leftJacobianSO3(omega);
  const Eigen::Matrix3d R0 = expSO3(omega);

  const double eps = 1e-7;
  Eigen::Matrix3d J_num;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d dw = Eigen::Vector3d::Zero();
    dw(i) = eps;
    const Eigen::Matrix3d R1 = expSO3(omega + dw);
    // Left convention: log(R1 · R0^T) ≈ J_l · δω
    const Eigen::Vector3d delta = logSO3(R1 * R0.transpose());
    J_num.col(i) = delta / eps;
  }
  EXPECT_NEAR((J - J_num).norm(), 0.0, 1e-4);
}

TEST(LeftJacobianSO3, NumericalLargeAngle) {
  // Test at a larger angle (~2.5 rad)
  const Eigen::Vector3d omega(1.5, -1.0, 1.2);
  const Eigen::Matrix3d J = leftJacobianSO3(omega);
  const Eigen::Matrix3d R0 = expSO3(omega);

  const double eps = 1e-7;
  Eigen::Matrix3d J_num;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d dw = Eigen::Vector3d::Zero();
    dw(i) = eps;
    const Eigen::Matrix3d R1 = expSO3(omega + dw);
    J_num.col(i) = logSO3(R1 * R0.transpose()) / eps;
  }
  EXPECT_NEAR((J - J_num).norm(), 0.0, 1e-4);
}

// ===========================================================================
// invLeftJacobianSO3()
// ===========================================================================

TEST(InvLeftJacobianSO3, IdentityAtZero) {
  const Eigen::Matrix3d Jinv = invLeftJacobianSO3(Eigen::Vector3d::Zero());
  EXPECT_NEAR((Jinv - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
}

TEST(InvLeftJacobianSO3, IsInverseOfLeftJacobian) {
  // J_l · J_l^{-1} = I  for several rotation vectors
  const std::vector<Eigen::Vector3d> omegas = {
      {0.5, -0.3, 0.8},
      {1.5, -1.0, 1.2},
      {0.01, 0.02, -0.015},
      {M_PI * 0.9, 0.0, 0.0}, // near π
  };
  for (const auto& omega : omegas) {
    const Eigen::Matrix3d J = leftJacobianSO3(omega);
    const Eigen::Matrix3d Jinv = invLeftJacobianSO3(omega);
    const Eigen::Matrix3d prod = J * Jinv;
    EXPECT_NEAR((prod - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolLoose)
        << "Failed for omega = " << omega.transpose();
  }
}

TEST(InvLeftJacobianSO3, NumericalComparison) {
  // Compare with explicitly inverting the left Jacobian
  const Eigen::Vector3d omega(0.7, -0.4, 1.1);
  const Eigen::Matrix3d J = leftJacobianSO3(omega);
  const Eigen::Matrix3d Jinv_direct = J.inverse();
  const Eigen::Matrix3d Jinv = invLeftJacobianSO3(omega);
  EXPECT_NEAR((Jinv - Jinv_direct).norm(), 0.0, kTolLoose);
}

// ===========================================================================
// gravity()
// ===========================================================================

TEST(Gravity, ENU) {
  const Eigen::Vector3d g = gravity();
  EXPECT_DOUBLE_EQ(g.x(), 0.0);
  EXPECT_DOUBLE_EQ(g.y(), 0.0);
  EXPECT_DOUBLE_EQ(g.z(), -9.81);
}
