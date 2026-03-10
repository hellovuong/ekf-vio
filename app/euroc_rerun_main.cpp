// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

// ============================================================================
//  Standalone EuRoC dataset runner with Rerun visualization
//
//  Usage:
//    euroc_rerun_runner <path_to_sequence> [config.yaml]
//                       [--connect [host:port]] [--save [file.rrd]]
//
//  Default: connects to a Rerun viewer at 127.0.0.1:9876
//
//  Docker — Linux (--network host):
//    Host:      rerun
//    Container: euroc_rerun_runner /data/MH_01_easy config/euroc.yaml
//
//  Docker — macOS / Windows (file-based):
//    Container: euroc_rerun_runner /data/MH_01_easy config/euroc.yaml \
//                                  --save /results/ekf_vio.rrd
//    Host:      rerun /results/ekf_vio.rrd
// ============================================================================

#include "ekf_vio/config.hpp"
#include "ekf_vio/ekf.hpp"
#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/logging.hpp"
#include "ekf_vio/stereo_rectifier.hpp"
#include "ekf_vio/stereo_tracker.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <rerun.hpp>
#include <string>
#include <vector>

using namespace ekf_vio;

int main(int argc, char** argv) {
  init_logging(spdlog::level::info);
  auto log = get_logger();

  if (argc < 2) {
    log->error(
        "Usage: {} <euroc_sequence_path> [config.yaml] [--connect [host:port]] [--save [file.rrd]]",
        argv[0]);
    return 1;
  }

  // ── Parse rerun output mode ─────────────────────────────────────────────
  std::string rerun_mode = "connect";
  std::string rerun_addr = "127.0.0.1:9876";
  std::string rerun_file = "ekf_vio.rrd";

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--connect") {
      rerun_mode = "connect";
      if (i + 1 < argc && argv[i + 1][0] != '-') rerun_addr = argv[++i];
    } else if (arg == "--save") {
      rerun_mode = "save";
      if (i + 1 < argc && argv[i + 1][0] != '-') rerun_file = argv[++i];
    }
  }

  // ── Load config ─────────────────────────────────────────────────────────
  const std::string config_path = (argc >= 3) ? argv[2] : "config/euroc.yaml";
  Config cfg;
  try {
    cfg = loadConfig(config_path);
    log->info("Loaded config from: {}", config_path);
  } catch (const std::exception& e) {
    log->error("Failed to load config '{}': {}", config_path, e.what());
    return 1;
  }

  // ── Load dataset ─────────────────────────────────────────────────────────
  EurocReader reader(argv[1]);
  if (!reader.load()) {
    log->error("Failed to load EuRoC sequence from: {}", argv[1]);
    return 1;
  }

  // ── Initialize Rerun ────────────────────────────────────────────────────
  rerun::RecordingStream rec("ekf_vio");
  rerun::Error err;
  if (rerun_mode == "save") {
    err = rec.save(rerun_file);
    log->info("Rerun: saving to '{}'", rerun_file);
  } else {
    err = rec.connect(rerun_addr);
    log->info("Rerun: connecting to {} (start 'rerun' on the host first)", rerun_addr);
  }

  if (err.is_err()) {
    log->error("Error while connect to rerun: Error Code {}: {}", static_cast<int>(err.code),
               err.description);
    return EXIT_FAILURE;
  }

  // EuRoC world frame: right-handed, Z up (ENU)
  rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

  // ── Set up VIO pipeline ─────────────────────────────────────────────────
  StereoRectifier rectifier;
  rectifier.init(cfg.camera);
  auto cam = makeStereoCamera(rectifier, cfg.camera);

  EKF ekf(cam, toNoiseParams(cfg.imu, cfg.ekf));
  StereoTracker tracker(cam, toTrackerParams(cfg.tracker));

  bool initialized = false;
  double last_imu_time = 0.0;
  int stereo_count = 0;

  // Accumulated trajectory points for continuous line strips
  std::vector<rerun::Vec3D> est_traj;
  std::vector<rerun::Vec3D> gt_traj;

  std::ofstream traj_out("euroc_traj.csv");
  traj_out << "# timestamp,px,py,pz,qw,qx,qy,qz\n";

  // ── Replay ──────────────────────────────────────────────────────────────
  reader.replay(
      [&](const ImuData& imu) {
        if (!initialized) {
          last_imu_time = imu.timestamp;
          return;
        }
        const double dt = imu.timestamp - last_imu_time;
        if (dt <= 0.0 || dt > 0.5) {
          last_imu_time = imu.timestamp;
          return;
        }
        ekf.predict(imu, dt);
        last_imu_time = imu.timestamp;
      },

      [&](const StereoImages& stereo) {
        if (stereo.left.empty() || stereo.right.empty()) return;

        // ── Initialisation ───────────────────────────────────────────────
        if (!initialized) {
          GroundTruth gt;
          if (reader.closestGroundTruth(stereo.timestamp, gt)) {
            ekf.state().T_wb = Sophus::SE3d(Sophus::SO3d(gt.q), gt.p);
            ekf.state().v = gt.v;
            ekf.state().b_g = gt.b_g;
            ekf.state().b_a = gt.b_a;
            log->info("Initialized from ground truth at t={:.6f}s", stereo.timestamp);
          } else {
            ekf.state().T_wb = Sophus::SE3d();
            ekf.state().v = Eigen::Vector3d::Zero();
            log->info("No ground truth; starting at identity");
          }
          const auto& ic = cfg.initial_covariance;
          ekf.state().P.setZero();
          ekf.state().P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * ic.position;
          ekf.state().P.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * ic.velocity;
          ekf.state().P.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * ic.orientation;
          ekf.state().P.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * ic.gyro_bias;
          ekf.state().P.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * ic.accel_bias;
          initialized = true;
          last_imu_time = stereo.timestamp;
          return;
        }

        // ── VIO step ─────────────────────────────────────────────────────
        cv::Mat rect_left;
        cv::Mat rect_right;
        rectifier.rectify(stereo.left, stereo.right, rect_left, rect_right);
        const auto features = tracker.track(rect_left, rect_right);
        if (!features.empty()) ekf.update(features);

        ++stereo_count;
        const State& s = ekf.state();
        const auto& p = s.T_wb.translation();
        const auto& q = s.T_wb.unit_quaternion();

        traj_out << std::fixed << std::setprecision(9) << stereo.timestamp << "," << p.x() << ","
                 << p.y() << "," << p.z() << "," << q.w() << "," << q.x() << "," << q.y() << ","
                 << q.z() << "\n";

        // ── Rerun logging ─────────────────────────────────────────────────
        rec.set_time_seconds("stable_time", stereo.timestamp);

        // Ground truth trajectory (green line strip)
        GroundTruth gt;
        if (reader.closestGroundTruth(stereo.timestamp, gt)) {
          gt_traj.emplace_back(std::array<float, 3>{static_cast<float>(gt.p.x()),
                                                    static_cast<float>(gt.p.y()),
                                                    static_cast<float>(gt.p.z())});
          rec.log("world/gt_trajectory",
                  rerun::LineStrips3D({gt_traj}).with_colors({rerun::Color(0, 200, 60)}));
        }

        // EKF estimated trajectory (red line strip)
        est_traj.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()),
                              static_cast<float>(p.z()));
        rec.log("world/est_trajectory",
                rerun::LineStrips3D({est_traj}).with_colors({rerun::Color(220, 50, 50)}));

        // T_{world←cam} = T_{world←body} * T_{body←cam}
        //               = T_wb            * T_cam_imu^{-1}
        // Twc = Twb * Tbc
        auto Twc = (s.T_wb * cam.T_cam_imu.inverse()).cast<float>();
        // Posed pinhole camera:
        rec.log("world/camera", rerun::Transform3D(rerun::Vec3D(Twc.translation().data()),
                                                   rerun::Mat3x3(Twc.so3().matrix().data())));

        rec.log("world/camera", rerun::Pinhole::from_focal_length_and_resolution(
                                    {static_cast<float>(cfg.camera.cam0.fx),
                                     static_cast<float>(cfg.camera.cam0.fy)},
                                    {static_cast<float>(cfg.camera.image_width),
                                     static_cast<float>(cfg.camera.image_height)}));

        // Left camera image (grayscale → RGB for Rerun display)
        cv::Mat rgb;
        cv::cvtColor(rect_left, rgb, cv::COLOR_GRAY2RGB);
        // Use rerun::borrow to create a view into the cv::Mat memory without copying it
        rec.log("world/camera",
                rerun::Image::from_rgb24(
                    rerun::borrow(rgb.data, rgb.total() * rgb.channels()),
                    {static_cast<uint32_t>(rgb.cols), static_cast<uint32_t>(rgb.rows)}));

        // Tracked 2D features overlaid on the camera image (yellow dots)
        if (!features.empty()) {
          std::vector<std::array<float, 3>> pts3d;
          pts3d.reserve(features.size());
          for (const auto& f : features) {
            const Eigen::Vector3d p_w = Twc.cast<double>() * f.p_c;
            pts3d.push_back({static_cast<float>(p_w.x()), static_cast<float>(p_w.y()),
                             static_cast<float>(p_w.z())});
          }
          rec.log(
              "world/landmarks",
              rerun::Points3D(pts3d).with_colors({rerun::Color(255, 200, 0)}).with_radii({0.01f}));
        }

        if (stereo_count % 100 == 0) {
          log->info("Frame {:4d}  t={:.3f}  feat={}  pos=({:.3f}, {:.3f}, {:.3f})", stereo_count,
                    stereo.timestamp, features.size(), p.x(), p.y(), p.z());
        }
      });

  const State& s = ekf.state();
  const auto& p_final = s.T_wb.translation();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames, {} IMU samples", stereo_count, reader.numImu());
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", p_final.x(), p_final.y(), p_final.z());
  log->info("Trajectory saved to euroc_traj.csv");
  return 0;
}
