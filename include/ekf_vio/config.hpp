// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

/**
 * ekf_vio/config.hpp — YAML configuration loader for EKF-VIO.
 *
 * Parses a standalone YAML file (config/euroc.yaml) into a Config struct
 * that feeds all sub-systems (camera, IMU, tracker, VO, EKF).
 */
#pragma once

#include "ekf_vio/ekf.hpp"
#include "ekf_vio/stereo_tracker.hpp"
#include "ekf_vio/stereo_vo.hpp"

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace ekf_vio {

// ── Sub-configs ──────────────────────────────────────────────

struct CameraIntrinsics {
  double fx, fy, cx, cy;
  std::vector<double> distortion;  // radtan [k1, k2, p1, p2]
};

struct CameraConfig {
  int image_width;
  int image_height;
  CameraIntrinsics cam0;
  CameraIntrinsics cam1;
  Eigen::Matrix4d T_BS0;  // sensor→body (EuRoC convention)
  Eigen::Matrix4d T_BS1;
};

struct ImuConfig {
  double noise_gyro;   // rad/s/√Hz
  double noise_accel;  // m/s²/√Hz
  double gyro_walk;    // rad/s²/√Hz
  double accel_walk;   // m/s³/√Hz
  double frequency;    // Hz
};

struct EkfConfig {
  double sigma_pixel;
  double vo_position_noise;
  double vo_orientation_noise;
  int landmark_max_age = 5;
};

struct TrackerConfig {
  int max_features;
  int fast_threshold;
  int lk_win_size;
  int lk_max_level;
  double min_disparity;
  double max_disparity;
  int stereo_search_radius = 50;
  double ransac_thresh_px;
  double epipolar_thresh_px = 3.0;
};

struct VoConfig {
  int min_pnp_points;
  double pnp_reproj_thresh;
  double kf_tracked_ratio;
  int kf_min_tracked;
  double max_translation_m;
  double max_rotation_deg;
};

struct InitialCovarianceConfig {
  double position;
  double velocity;
  double orientation;
  double gyro_bias;
  double accel_bias;
};

struct Config {
  CameraConfig camera;
  ImuConfig imu;
  EkfConfig ekf;
  TrackerConfig tracker;
  VoConfig vo;
  InitialCovarianceConfig initial_covariance;
};

// ── Helpers ──────────────────────────────────────────────────

namespace detail {

inline CameraIntrinsics loadIntrinsics(const YAML::Node& n) {
  CameraIntrinsics ci;
  ci.fx = n["fx"].as<double>();
  ci.fy = n["fy"].as<double>();
  ci.cx = n["cx"].as<double>();
  ci.cy = n["cy"].as<double>();
  ci.distortion = n["distortion"].as<std::vector<double>>();
  return ci;
}

inline Eigen::Matrix4d loadMatrix4d(const YAML::Node& n) {
  auto v = n.as<std::vector<double>>();
  Eigen::Matrix4d M;
  for (int i = 0; i < 16; ++i)
    M(i / 4, i % 4) = v[static_cast<size_t>(i)];
  return M;
}

}  // namespace detail

// ── Main loader ──────────────────────────────────────────────

inline Config loadConfig(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  Config cfg;

  // Camera
  const auto& cam = root["camera"];
  cfg.camera.image_width = cam["image_width"].as<int>();
  cfg.camera.image_height = cam["image_height"].as<int>();
  cfg.camera.cam0 = detail::loadIntrinsics(cam["cam0"]);
  cfg.camera.cam1 = detail::loadIntrinsics(cam["cam1"]);
  cfg.camera.T_BS0 = detail::loadMatrix4d(cam["T_BS0"]);
  cfg.camera.T_BS1 = detail::loadMatrix4d(cam["T_BS1"]);

  // IMU
  const auto& imu = root["imu"];
  cfg.imu.noise_gyro = imu["noise_gyro"].as<double>();
  cfg.imu.noise_accel = imu["noise_accel"].as<double>();
  cfg.imu.gyro_walk = imu["gyro_walk"].as<double>();
  cfg.imu.accel_walk = imu["accel_walk"].as<double>();
  cfg.imu.frequency = imu["frequency"].as<double>();

  // EKF
  const auto& ekf = root["ekf"];
  cfg.ekf.sigma_pixel = ekf["sigma_pixel"].as<double>();
  cfg.ekf.vo_position_noise = ekf["vo_position_noise"].as<double>();
  cfg.ekf.vo_orientation_noise = ekf["vo_orientation_noise"].as<double>();
  if (ekf["landmark_max_age"]) cfg.ekf.landmark_max_age = ekf["landmark_max_age"].as<int>();

  // Tracker
  const auto& tr = root["tracker"];
  cfg.tracker.max_features = tr["max_features"].as<int>();
  cfg.tracker.fast_threshold = tr["fast_threshold"].as<int>();
  cfg.tracker.lk_win_size = tr["lk_win_size"].as<int>();
  cfg.tracker.lk_max_level = tr["lk_max_level"].as<int>();
  cfg.tracker.min_disparity = tr["min_disparity"].as<double>();
  cfg.tracker.max_disparity = tr["max_disparity"].as<double>();
  cfg.tracker.ransac_thresh_px = tr["ransac_thresh_px"].as<double>();
  if (tr["stereo_search_radius"])
    cfg.tracker.stereo_search_radius = tr["stereo_search_radius"].as<int>();
  if (tr["epipolar_thresh_px"])
    cfg.tracker.epipolar_thresh_px = tr["epipolar_thresh_px"].as<double>();

  // VO
  const auto& vo = root["vo"];
  cfg.vo.min_pnp_points = vo["min_pnp_points"].as<int>();
  cfg.vo.pnp_reproj_thresh = vo["pnp_reproj_thresh"].as<double>();
  cfg.vo.kf_tracked_ratio = vo["kf_tracked_ratio"].as<double>();
  cfg.vo.kf_min_tracked = vo["kf_min_tracked"].as<int>();
  cfg.vo.max_translation_m = vo["max_translation_m"].as<double>();
  cfg.vo.max_rotation_deg = vo["max_rotation_deg"].as<double>();

  // Initial covariance
  const auto& ic = root["initial_covariance"];
  cfg.initial_covariance.position = ic["position"].as<double>();
  cfg.initial_covariance.velocity = ic["velocity"].as<double>();
  cfg.initial_covariance.orientation = ic["orientation"].as<double>();
  cfg.initial_covariance.gyro_bias = ic["gyro_bias"].as<double>();
  cfg.initial_covariance.accel_bias = ic["accel_bias"].as<double>();

  return cfg;
}

// ── Config → Params factories ────────────────────────────────

inline StereoTracker::Params toTrackerParams(const TrackerConfig& tc) {
  StereoTracker::Params p;
  p.max_features = tc.max_features;
  p.fast_threshold = tc.fast_threshold;
  p.lk_win_size = tc.lk_win_size;
  p.lk_max_level = tc.lk_max_level;
  p.min_disparity = tc.min_disparity;
  p.max_disparity = tc.max_disparity;
  p.stereo_search_radius = tc.stereo_search_radius;
  p.ransac_thresh_px = tc.ransac_thresh_px;
  p.epipolar_thresh_px = tc.epipolar_thresh_px;
  return p;
}

inline StereoVO::Params toVoParams(const VoConfig& vc) {
  StereoVO::Params p;
  p.min_pnp_points = vc.min_pnp_points;
  p.pnp_reproj_thresh = vc.pnp_reproj_thresh;
  p.kf_tracked_ratio = vc.kf_tracked_ratio;
  p.kf_min_tracked = vc.kf_min_tracked;
  p.max_translation_m = vc.max_translation_m;
  p.max_rotation_deg = vc.max_rotation_deg;
  return p;
}

inline EKF::NoiseParams toNoiseParams(const ImuConfig& imu, const EkfConfig& ekf) {
  EKF::NoiseParams n;
  n.sigma_gyro = imu.noise_gyro;
  n.sigma_accel = imu.noise_accel;
  n.sigma_gyro_bias = imu.gyro_walk;
  n.sigma_accel_bias = imu.accel_walk;
  n.sigma_pixel = ekf.sigma_pixel;
  n.landmark_max_age = ekf.landmark_max_age;
  return n;
}

}  // namespace ekf_vio
