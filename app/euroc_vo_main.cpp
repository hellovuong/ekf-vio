// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

// ============================================================================
//  Standalone EuRoC dataset runner for Keyframe-based Stereo VO (no IMU)
//
//  Usage:  ./euroc_vo_runner <path_to_sequence> [config.yaml]
// ============================================================================

#include "ekf_vio/config.hpp"
#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/logging.hpp"
#include "ekf_vio/stereo_rectifier.hpp"
#include "ekf_vio/stereo_tracker.hpp"
#include "ekf_vio/stereo_vo.hpp"

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

  // Stereo rectification + camera setup (VO-only, no T_cam_imu)
  StereoRectifier rectifier;
  rectifier.init(cfg.camera);
  auto cam = makeStereoCamera(rectifier);

  // Initialise subsystems from config
  StereoTracker tracker(cam, toTrackerParams(cfg.tracker));
  StereoVO vo(cam, toVoParams(cfg.vo));

  bool have_gt = false;
  int frame_count = 0;

  std::ofstream traj_out("euroc_traj.csv");
  traj_out << "# timestamp,px,py,pz,qw,qx,qy,qz\n";

  // Replay — only process stereo frames (ignore IMU)
  reader.replay([](const ImuData&) {},

                [&](const StereoImages& stereo) {
                  if (stereo.left.empty() || stereo.right.empty()) return;

                  if (frame_count == 0 && !have_gt) {
                    GroundTruth gt;
                    if (reader.closestGroundTruth(stereo.timestamp, gt)) {
                      // GT available for evaluation alignment — VO starts at identity
                    }
                    have_gt = true;
                  }

                  cv::Mat rect_left;
                  cv::Mat rect_right;
                  rectifier.rectify(stereo.left, stereo.right, rect_left, rect_right);
                  const auto features = tracker.track(rect_left, rect_right);
                  Eigen::Isometry3d T_wc = vo.process(features);

                  ++frame_count;
                  const Eigen::Vector3d p = T_wc.translation();
                  const Eigen::Quaterniond q(T_wc.rotation());

                  traj_out << std::fixed << std::setprecision(9) << stereo.timestamp << "," << p.x()
                           << "," << p.y() << "," << p.z() << "," << q.w() << "," << q.x() << ","
                           << q.y() << "," << q.z() << "\n";

                  if (frame_count % 100 == 0) {
                    log->info(
                        "Frame {:4d}  t={:.3f}  feat={}  inliers={}  kf_lm={}"
                        "  pos=({:.3f}, {:.3f}, {:.3f})",
                        frame_count, stereo.timestamp, features.size(), vo.numInliers(),
                        vo.numKeyframeLandmarks(), p.x(), p.y(), p.z());
                  }
                });

  const Eigen::Vector3d p = vo.pose().translation();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames", frame_count);
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", p.x(), p.y(), p.z());
  log->info("Trajectory saved to euroc_traj.csv");
  return 0;
}
