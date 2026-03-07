// ============================================================================
//  Standalone EuRoC dataset runner for EKF-VIO
//
//  Usage:  ./euroc_runner <path_to_sequence> [euroc.yaml]
//          e.g.  ./euroc_runner /data/EuRoC/MH_01_easy
// ============================================================================

#include "ekf_vio/ekf.hpp"
#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/stereo_tracker.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace ekf_vio;

// Build camera & noise config (hard-coded EuRoC defaults — override via yaml
// later)
static StereoCamera makeEurocCamera() {
  StereoCamera cam;
  cam.fx = 458.654;
  cam.fy = 457.296;
  cam.cx = 367.215;
  cam.cy = 248.375;
  cam.baseline = 0.11007;

  Eigen::Matrix4d T;
  T << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
      0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
      -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949, 0.0,
      0.0, 0.0, 1.0;
  cam.T_cam_imu = Eigen::Isometry3d(T);
  return cam;
}

static EKF::NoiseParams makeEurocNoise() {
  EKF::NoiseParams n;
  n.sigma_gyro = 1.6968e-4;
  n.sigma_accel = 2.0000e-3;
  n.sigma_gyro_bias = 1.9393e-5;
  n.sigma_accel_bias = 3.0000e-5;
  n.sigma_pixel = 1.5;
  return n;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <euroc_sequence_path>\n"
              << "  e.g.  " << argv[0] << " /data/EuRoC/MH_01_easy\n";
    return 1;
  }

  // ------------------------------------------------------------------
  // Load dataset
  // ------------------------------------------------------------------
  EurocReader reader(argv[1]);
  if (!reader.load()) {
    std::cerr << "Failed to load EuRoC sequence from: " << argv[1] << "\n";
    return 1;
  }

  // ------------------------------------------------------------------
  // Initialise EKF + tracker
  // ------------------------------------------------------------------
  auto cam = makeEurocCamera();
  auto noise = makeEurocNoise();

  EKF ekf(cam, noise);
  StereoTracker tracker(cam, std::move(StereoTracker::Params{}));

  bool initialized = false;
  double last_imu_time = 0.0;
  int stereo_count = 0;

  // Optional: open output file for trajectory
  std::ofstream traj_out("euroc_traj.csv");
  traj_out << "# timestamp,px,py,pz,qw,qx,qy,qz\n";

  // ------------------------------------------------------------------
  // Replay dataset in chronological order
  // ------------------------------------------------------------------
  reader.replay(
      // --- IMU callback ---
      [&](const ImuData &imu) {
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

      // --- Stereo callback ---
      [&](const StereoImages &stereo) {
        if (stereo.left.empty() || stereo.right.empty())
          return;

        if (!initialized) {
          // Initialise attitude from ground truth if available,
          // otherwise start at identity
          GroundTruth gt;
          if (reader.closestGroundTruth(stereo.timestamp, gt)) {
            ekf.state().q = gt.q;
            ekf.state().p = gt.p;
            ekf.state().v = gt.v;
            ekf.state().b_g = gt.b_g;
            ekf.state().b_a = gt.b_a;
            std::cout << "[Init] Using ground truth at t=" << std::fixed
                      << std::setprecision(6) << stereo.timestamp << "s\n";
          } else {
            ekf.state().q = Eigen::Quaterniond::Identity();
            ekf.state().p = Eigen::Vector3d::Zero();
            ekf.state().v = Eigen::Vector3d::Zero();
            std::cout << "[Init] No ground truth; starting at identity.\n";
          }
          initialized = true;
          last_imu_time = stereo.timestamp;
          return; // skip tracking on first frame
        }

        const auto features = tracker.track(stereo.left, stereo.right);

        if (!features.empty()) {
          ekf.update(features);
        }

        ++stereo_count;

        // Write trajectory
        const State &s = ekf.state();
        traj_out << std::fixed << std::setprecision(9) << stereo.timestamp
                 << "," << s.p.x() << "," << s.p.y() << "," << s.p.z() << ","
                 << s.q.w() << "," << s.q.x() << "," << s.q.y() << ","
                 << s.q.z() << "\n";

        // Print progress every 100 frames
        if (stereo_count % 100 == 0) {
          std::cout << "[Frame " << stereo_count << "] t=" << std::fixed
                    << std::setprecision(3) << stereo.timestamp << "  pos=("
                    << std::setprecision(3) << s.p.x() << ", " << s.p.y()
                    << ", " << s.p.z() << ")\n";
        }
      });

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------
  const State &s = ekf.state();
  std::cout << "\n=== Done ===\n"
            << "Processed " << stereo_count << " stereo frames, "
            << reader.numImu() << " IMU samples.\n"
            << "Final pos: (" << s.p.x() << ", " << s.p.y() << ", " << s.p.z()
            << ")\n"
            << "Trajectory saved to euroc_traj.csv\n";

  return 0;
}
