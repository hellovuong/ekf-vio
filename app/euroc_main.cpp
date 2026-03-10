// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

// ============================================================================
//  Standalone EuRoC dataset runner for EKF-VIO
//
//  Usage:  ./euroc_runner <path_to_sequence> [config.yaml]
// ============================================================================

#include "ekf_vio/config.hpp"
#include "ekf_vio/ekf.hpp"
#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/logging.hpp"
#include "ekf_vio/stereo_rectifier.hpp"
#include "ekf_vio/stereo_tracker.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>

using namespace ekf_vio;

int main(int argc, char** argv) {
  init_logging(spdlog::level::info);
  auto log = get_logger();

  if (argc < 2) {
    log->error("Usage: {} <euroc_sequence_path> [config.yaml]", argv[0]);
    return 1;
  }

  // Load config
  std::string config_path = (argc >= 3) ? argv[2] : "config/euroc.yaml";
  Config cfg;
  try {
    cfg = loadConfig(config_path);
    log->info("Loaded config from: {}", config_path);
  } catch (const std::exception& e) {
    log->error("Failed to load config '{}': {}", config_path, e.what());
    return 1;
  }

  // Load dataset
  EurocReader reader(argv[1]);
  if (!reader.load()) {
    log->error("Failed to load EuRoC sequence from: {}", argv[1]);
    return 1;
  }

  // Stereo rectification + camera setup
  StereoRectifier rectifier;
  rectifier.init(cfg.camera);
  auto cam = makeStereoCamera(rectifier, cfg.camera);

  // Initialise subsystems from config
  EKF ekf(cam, toNoiseParams(cfg.imu, cfg.ekf));
  StereoTracker tracker(cam, toTrackerParams(cfg.tracker));

  bool initialized = false;
  double last_imu_time = 0.0;
  int stereo_count = 0;

  // Per-stage timing accumulators (microseconds)
  using Clock = std::chrono::steady_clock;
  double t_rectify_us = 0.0;
  double t_track_us = 0.0;
  double t_update_us = 0.0;
  double t_total_us = 0.0;

  auto print_timing = [&]() {
    const auto n = static_cast<double>(stereo_count);
    log->info("── Timing ({} frames) ──────────────────────────────────", stereo_count);
    log->info("  Rectify : {:6.2f} ms/frame", t_rectify_us / n / 1e3);
    log->info("  Track   : {:6.2f} ms/frame", t_track_us / n / 1e3);
    log->info("  EKF upd : {:6.2f} ms/frame", t_update_us / n / 1e3);
    log->info("  Total   : {:6.2f} ms/frame  →  {:.1f} Hz", t_total_us / n / 1e3,
              n * 1e6 / t_total_us);
  };

  std::ofstream traj_out("euroc_traj.csv");
  traj_out << "# timestamp,px,py,pz,qw,qx,qy,qz\n";

  // Replay dataset
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

        if (!initialized) {
          GroundTruth gt;
          if (reader.closestGroundTruth(stereo.timestamp, gt)) {
            ekf.state().T_wb = Sophus::SE3d(Sophus::SO3d(gt.q), gt.p);
            ekf.state().v = gt.v;
            ekf.state().b_g = gt.b_g;
            ekf.state().b_a = gt.b_a;
            log->info("Using ground truth at t={:.6f}s", stereo.timestamp);
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

        const auto t0 = Clock::now();

        cv::Mat rect_left;
        cv::Mat rect_right;
        rectifier.rectify(stereo.left, stereo.right, rect_left, rect_right);
        const auto t1 = Clock::now();

        const auto features = tracker.track(rect_left, rect_right);
        const auto t2 = Clock::now();

        if (!features.empty()) ekf.update(features);
        const auto t3 = Clock::now();

        t_rectify_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        t_track_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        t_update_us += std::chrono::duration<double, std::micro>(t3 - t2).count();
        t_total_us += std::chrono::duration<double, std::micro>(t3 - t0).count();

        ++stereo_count;
        const State& s = ekf.state();
        const auto& p = s.T_wb.translation();
        const auto& q = s.T_wb.unit_quaternion();
        traj_out << std::fixed << std::setprecision(9) << stereo.timestamp << "," << p.x() << ","
                 << p.y() << "," << p.z() << "," << q.w() << "," << q.x() << "," << q.y() << ","
                 << q.z() << "\n";

        if (stereo_count % 100 == 0) {
          log->info("Frame {:4d}  t={:.3f}  feat={}  pos=({:.3f}, {:.3f}, {:.3f})", stereo_count,
                    stereo.timestamp, features.size(), p.x(), p.y(), p.z());
          print_timing();
        }
      });

  print_timing();

  const State& s = ekf.state();
  const auto& p_final = s.T_wb.translation();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames, {} IMU samples", stereo_count, reader.numImu());
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", p_final.x(), p_final.y(), p_final.z());
  log->info("Trajectory saved to euroc_traj.csv");
  return 0;
}
