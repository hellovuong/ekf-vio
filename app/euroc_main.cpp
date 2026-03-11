// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

// ============================================================================
//  Standalone EuRoC dataset runner for EKF-VIO
//
//  Usage:  ./euroc_runner <path_to_sequence> [config.yaml] [--connect [host:port]]
//  [--save-rerun-file [file.rrd]] [--debug]
//
//  Without Rerun flags: runs headless with per-stage timing.
//  With --connect / --save-rerun-file: streams visualisation to Rerun (requires -DWITH_RERUN=ON).
//  With --debug: sets logger to debug level; also logs per-frame track/stereo debug images to
//  Rerun (debug/left_track, debug/stereo_right) when a Rerun flag is also given.
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
#include <pthread.h>
#include <string>

#ifdef EKF_VIO_WITH_RERUN
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <rerun.hpp>
#include <thread>
#include <vector>

// Data package sent from the VIO thread to the Rerun logging thread.
struct RerunLogData {
  double timestamp{0.0};
  Eigen::Vector3d est_pos;
  bool has_gt{false};
  Eigen::Vector3d gt_pos;
  Sophus::SE3f twc;
  float fx{0.f}, fy{0.f}, img_w{0.f}, img_h{0.f};
  cv::Mat rect_left;                     // grayscale; shallow-copy is safe (ref-counted)
  std::vector<Eigen::Vector3d> pts_cam;  // raw p_c in camera frame; transform done in log thread

  // Debug-mode fields (populated only when --debug is active)
  bool debug_mode{false};
  cv::Mat rect_right;                           // right rectified image
  std::vector<std::array<float, 2>> left_kps;   // tracked feature positions on left image
  std::vector<std::array<float, 2>> right_kps;  // stereo match positions on right image
};
#endif  // EKF_VIO_WITH_RERUN

using namespace ekf_vio;

int main(int argc, char** argv) {
  pthread_setname_np(pthread_self(), "vio_main");
  init_logging(spdlog::level::info);
  auto log = get_logger();

  if (argc < 2) {
    log->error(
        "Usage: {} <euroc_sequence_path> [config.yaml] [--connect [host:port]] [--save-rerun-file "
        "[file.rrd]] [--debug]",
        argv[0]);
    return 1;
  }

  // ── Parse optional flags ─────────────────────────────────────────────────
  bool want_rerun = false;
  bool debug_mode = false;
  std::string rerun_mode = "connect";
  std::string rerun_addr = "127.0.0.1:9876";
  std::string rerun_file = "ekf_vio.rrd";

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--connect") {
      want_rerun = true;
      rerun_mode = "connect";
      if (i + 1 < argc && argv[i + 1][0] != '-') rerun_addr = argv[++i];
    } else if (arg == "--save-rerun-file") {
      want_rerun = true;
      rerun_mode = "save";
      if (i + 1 < argc && argv[i + 1][0] != '-') rerun_file = argv[++i];
    } else if (arg == "--debug") {
      debug_mode = true;
    }
  }

  if (debug_mode) {
    log->set_level(spdlog::level::debug);
    log->debug("Debug mode enabled");
  }

#ifndef EKF_VIO_WITH_RERUN
  if (want_rerun) {
    log->error(
        "Rerun visualisation requested but this binary was built without Rerun support.\n"
        "Rebuild with:  colcon build --cmake-args -DWITH_RERUN=ON");
    return 1;
  }
#endif

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

  // ── Stereo rectification + camera setup ─────────────────────────────────
  StereoRectifier rectifier;
  rectifier.init(cfg.camera);
  auto cam = makeStereoCamera(rectifier, cfg.camera);

  // ── Initialise VIO subsystems ────────────────────────────────────────────
  EKF ekf(cam, toNoiseParams(cfg.imu, cfg.ekf));
  StereoTracker tracker(cam, toTrackerParams(cfg.tracker));
  auto tracker_params = toTrackerParams(cfg.tracker);
  if (debug_mode) {
    tracker_params.debug_save_dir = "debug_frames";
    tracker_params.debug_save_start_frame = 0;
    tracker_params.debug_save_count = 5;
    log->debug("Tracker debug frames will be saved to: debug_frames/");
  }
  StereoTracker tracker(cam, tracker_params);

  bool initialized = false;
  double last_imu_time = 0.0;
  int stereo_count = 0;

  // ── Per-stage timing accumulators (microseconds) ─────────────────────────
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

#ifdef EKF_VIO_WITH_RERUN
  // ── Initialise Rerun (when requested) ───────────────────────────────────
  rerun::RecordingStream rec("ekf_vio");
  if (want_rerun) {
    rerun::Error err;
    if (rerun_mode == "save") {
      err = rec.save(rerun_file);
      log->info("Rerun: saving to '{}'", rerun_file);
    } else {
      err = rec.connect(rerun_addr);
      log->info("Rerun: connecting to {} (start 'rerun' on the host first)", rerun_addr);
    }
    if (err.is_err()) {
      log->error("Rerun init failed (code {}): {}", static_cast<int>(err.code), err.description);
      return 1;
    }
    rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  }

  // ── Async Rerun logging thread ───────────────────────────────────────────
  // The VIO callback packages data and pushes it; the logging thread drains
  // it asynchronously so Rerun I/O never stalls the EKF.
  constexpr size_t kMaxQueueSize = 8;
  std::queue<RerunLogData> log_queue;
  std::mutex log_mutex;
  std::condition_variable log_cv;
  std::atomic<bool> log_stop{false};

  auto T_imu_cam = cam.T_cam_imu.inverse().cast<float>();

  // Thread is started only when want_rerun is true.
  std::thread log_thread;
  if (want_rerun) {
    log_thread = std::thread([&]() {
      pthread_setname_np(pthread_self(), "rerun_log");
      std::vector<rerun::Vec3D> est_traj;

      while (true) {
        RerunLogData data;
        {
          std::unique_lock<std::mutex> lock(log_mutex);
          log_cv.wait(lock, [&] { return !log_queue.empty() || log_stop.load(); });
          if (log_queue.empty()) break;
          data = std::move(log_queue.front());
          log_queue.pop();
        }

        rec.set_time_seconds("stable_time", data.timestamp);

        // Estimated trajectory (red line strip)
        est_traj.emplace_back(std::array<float, 3>{static_cast<float>(data.est_pos.x()),
                                                   static_cast<float>(data.est_pos.y()),
                                                   static_cast<float>(data.est_pos.z())});
        rec.log("world/est_trajectory", rerun::LineStrips3D(est_traj)
                                            .with_colors({rerun::Color(220, 50, 50)})
                                            .with_radii({0.01f}));

        // Camera pose + pinhole
        rec.log("world/camera", rerun::Transform3D(rerun::Vec3D(data.twc.translation().data()),
                                                   rerun::Mat3x3(data.twc.so3().matrix().data())));
        rec.log("world/camera", rerun::Pinhole::from_focal_length_and_resolution(
                                    {data.fx, data.fy}, {data.img_w, data.img_h}));

        // Camera image (colour conversion off the VIO critical path)
        cv::Mat rgb;
        cv::cvtColor(data.rect_left, rgb, cv::COLOR_GRAY2RGB);
        rec.log("world/camera",
                rerun::Image::from_rgb24(
                    rerun::borrow(rgb.data, rgb.total() * rgb.channels()),
                    {static_cast<uint32_t>(rgb.cols), static_cast<uint32_t>(rgb.rows)}));

        // 3D landmarks (yellow dots) — camera→world transform done here
        if (!data.pts_cam.empty()) {
          std::vector<std::array<float, 3>> pts3d;
          pts3d.reserve(data.pts_cam.size());
          for (const auto& pc : data.pts_cam) {
            const Eigen::Vector3f p_w = data.twc * pc.cast<float>();
            pts3d.push_back({p_w.x(), p_w.y(), p_w.z()});
          }
          rec.log(
              "world/landmarks",
              rerun::Points3D(pts3d).with_colors({rerun::Color(255, 200, 0)}).with_radii({0.01f}));
        }

        // Debug: per-frame tracked features and stereo matches
        if (data.debug_mode) {
          // Left image with tracked feature overlay (green dots)
          cv::Mat rgb_left;
          cv::cvtColor(data.rect_left, rgb_left, cv::COLOR_GRAY2RGB);
          rec.log(
              "debug/left_track",
              rerun::Image::from_rgb24(
                  rerun::borrow(rgb_left.data, rgb_left.total() * rgb_left.channels()),
                  {static_cast<uint32_t>(rgb_left.cols), static_cast<uint32_t>(rgb_left.rows)}));
          if (!data.left_kps.empty()) {
            rec.log("debug/left_track", rerun::Points2D(data.left_kps)
                                            .with_colors({rerun::Color(0, 230, 0)})
                                            .with_radii({3.0f}));
          }

          // Right image with stereo match overlay (orange dots)
          if (!data.rect_right.empty()) {
            cv::Mat rgb_right;
            cv::cvtColor(data.rect_right, rgb_right, cv::COLOR_GRAY2RGB);
            rec.log("debug/stereo_right",
                    rerun::Image::from_rgb24(
                        rerun::borrow(rgb_right.data, rgb_right.total() * rgb_right.channels()),
                        {static_cast<uint32_t>(rgb_right.cols),
                         static_cast<uint32_t>(rgb_right.rows)}));
            if (!data.right_kps.empty()) {
              rec.log("debug/stereo_right", rerun::Points2D(data.right_kps)
                                                .with_colors({rerun::Color(255, 140, 0)})
                                                .with_radii({3.0f}));
            }
          }
        }
      }
    });
  }
#endif  // EKF_VIO_WITH_RERUN

  // ── Replay ───────────────────────────────────────────────────────────────
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

        // ── VIO step ─────────────────────────────────────────────────────
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

#ifdef EKF_VIO_WITH_RERUN
        // ── Push data to Rerun logging thread ────────────────────────────
        if (want_rerun) {
          auto Twc = s.T_wb.cast<float>() * T_imu_cam;

          RerunLogData log_data;
          log_data.timestamp = stereo.timestamp;
          log_data.est_pos = p;
          log_data.twc = Twc;
          log_data.fx = static_cast<float>(cfg.camera.cam0.fx);
          log_data.fy = static_cast<float>(cfg.camera.cam0.fy);
          log_data.img_w = static_cast<float>(cfg.camera.image_width);
          log_data.img_h = static_cast<float>(cfg.camera.image_height);
          log_data.rect_left = rect_left;  // shallow copy; ref-count keeps data alive

          GroundTruth gt;
          if (reader.closestGroundTruth(stereo.timestamp, gt)) {
            log_data.has_gt = true;
            log_data.gt_pos = gt.p;
          }
          if (!features.empty()) {
            log_data.pts_cam.reserve(features.size());
            for (const auto& f : features)
              log_data.pts_cam.push_back(f.p_c);
          }

          if (debug_mode) {
            log_data.debug_mode = true;
            log_data.rect_right = rect_right;  // shallow copy; ref-counted
            log_data.left_kps.reserve(features.size());
            log_data.right_kps.reserve(features.size());
            for (const auto& f : features) {
              log_data.left_kps.push_back({static_cast<float>(f.u_l), static_cast<float>(f.v_l)});
              log_data.right_kps.push_back({static_cast<float>(f.u_r), static_cast<float>(f.v_r)});
            }
          }

          {
            const std::lock_guard<std::mutex> lock(log_mutex);
            if (log_queue.size() < kMaxQueueSize) {
              log_queue.push(std::move(log_data));
              log_cv.notify_one();
            }
            // Queue full → drop frame (visualisation is non-critical)
          }
        }
#endif  // EKF_VIO_WITH_RERUN

        if (stereo_count % 100 == 0) {
          log->info("Frame {:4d}  t={:.3f}  feat={}  pos=({:.3f}, {:.3f}, {:.3f})", stereo_count,
                    stereo.timestamp, features.size(), p.x(), p.y(), p.z());
          print_timing();
        }
      });

#ifdef EKF_VIO_WITH_RERUN
  if (log_thread.joinable()) {
    log_stop = true;
    log_cv.notify_one();
    log_thread.join();
  }
#endif  // EKF_VIO_WITH_RERUN

  print_timing();

  const State& s = ekf.state();
  const auto& p_final = s.T_wb.translation();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames, {} IMU samples", stereo_count, reader.numImu());
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", p_final.x(), p_final.y(), p_final.z());
  log->info("Trajectory saved to euroc_traj.csv");
  return 0;
}
