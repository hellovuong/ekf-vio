// ============================================================================
//  Standalone EuRoC dataset runner for EKF-VIO
//
//  Usage:  ./euroc_runner <path_to_sequence> [euroc.yaml]
//          e.g.  ./euroc_runner /data/EuRoC/MH_01_easy
// ============================================================================

#include "ekf_vio/ekf.hpp"
#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/logging.hpp"
#include "ekf_vio/stereo_tracker.hpp"
#include "ekf_vio/stereo_vo.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace ekf_vio;

// ============================================================================
// Stereo rectification for EuRoC raw images
// ============================================================================
struct StereoRectifier {
  cv::Mat map1_left, map2_left;
  cv::Mat map1_right, map2_right;
  double fx, fy, cx, cy, baseline;
  Eigen::Matrix3d R_rect;        // rectification rotation for left cam (R1)

  void init() {
    // cam0 (left) intrinsics & distortion
    cv::Mat K0 = (cv::Mat_<double>(3, 3) <<
        458.654, 0.0, 367.215,
        0.0, 457.296, 248.375,
        0.0, 0.0, 1.0);
    cv::Mat D0 = (cv::Mat_<double>(4, 1) <<
        -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    // cam1 (right) intrinsics & distortion
    cv::Mat K1 = (cv::Mat_<double>(3, 3) <<
        457.587, 0.0, 379.999,
        0.0, 456.134, 255.238,
        0.0, 0.0, 1.0);
    cv::Mat D1 = (cv::Mat_<double>(4, 1) <<
        -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);

    // T_BS0 and T_BS1 from EuRoC sensor.yaml.
    // EuRoC convention: T_BS = sensor → body.
    Eigen::Matrix4d T0, T1;
    T0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
          0.999557249008,   0.0149672133247, 0.025715529948,  -0.064676986768,
         -0.0257744366974,  0.00375618835797, 0.999660727178,  0.00981073058949,
          0.0, 0.0, 0.0, 1.0;
    T1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
          0.999598781151,   0.0130119051815, 0.0251588363115,  0.0453689425024,
         -0.0253898008918,  0.0179005838253, 0.999517347078,   0.00786212447038,
          0.0, 0.0, 0.0, 1.0;

    // stereoRectify expects R,T = T_{second <- first} = T_{cam1 <- cam0}.
    // With EuRoC T_BS = sensor→body: T_{cam1<-cam0} = inv(T_BS1) * T_BS0
    Eigen::Matrix4d T_rel = T1.inverse() * T0;
    cv::Mat R_cv(3, 3, CV_64F), T_cv(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        R_cv.at<double>(r, c) = T_rel(r, c);
      T_cv.at<double>(r) = T_rel(r, 3);
    }

    cv::Size imgSize(752, 480);
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K0, D0, K1, D1, imgSize, R_cv, T_cv,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0, imgSize);

    cv::initUndistortRectifyMap(K0, D0, R1, P1, imgSize, CV_32FC1,
                                map1_left, map2_left);
    cv::initUndistortRectifyMap(K1, D1, R2, P2, imgSize, CV_32FC1,
                                map1_right, map2_right);

    // Store R1 for T_cam_imu correction
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R_rect(r, c) = R1.at<double>(r, c);

    double f_rect = P1.at<double>(0, 0);  // fx == fy after stereoRectify
    fx = f_rect;
    fy = f_rect;
    cx = P1.at<double>(0, 2);
    cy = P1.at<double>(1, 2);
    baseline = std::abs(P2.at<double>(0, 3)) / f_rect;

    ekf_vio::get_logger()->info("Rectify: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f} baseline={:.4f}m",
                                 fx, fy, cx, cy, baseline);
  }

  void rectify(const cv::Mat &raw_left, const cv::Mat &raw_right,
               cv::Mat &rect_left, cv::Mat &rect_right) const {
    cv::remap(raw_left, rect_left, map1_left, map2_left, cv::INTER_LINEAR);
    cv::remap(raw_right, rect_right, map1_right, map2_right, cv::INTER_LINEAR);
  }
};

// Build camera config using rectified intrinsics
static StereoCamera makeEurocCamera(const StereoRectifier &rect) {
  StereoCamera cam;
  cam.fx = rect.fx;
  cam.fy = rect.fy;
  cam.cx = rect.cx;
  cam.cy = rect.cy;
  cam.baseline = rect.baseline;

  // T_{cam←imu}
  // T_BS0 from EuRoC is T_{body←cam} (sensor→body convention).
  // We need T_{cam←body} = T_BS0^{-1}.
  // The rectified camera frame is rotated by R1 relative to the original:
  //   T_{rect_cam←imu} = R1 * T_{cam←imu} = R1 * T_BS0^{-1}
  Eigen::Matrix4d T_bs0;
  T_bs0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
      0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
      -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949, 0.0,
      0.0, 0.0, 1.0;
  Eigen::Matrix4d T_rect = Eigen::Matrix4d::Identity();
  T_rect.block<3,3>(0,0) = rect.R_rect;
  cam.T_cam_imu = Eigen::Isometry3d(T_rect * T_bs0.inverse());

  return cam;
}

static EKF::NoiseParams makeEurocNoise() {
  EKF::NoiseParams n;
  n.sigma_gyro = 1.6968e-4;
  n.sigma_accel = 2.0000e-3;
  n.sigma_gyro_bias = 1.9393e-5;
  n.sigma_accel_bias = 3.0000e-3;
  n.sigma_pixel = 1.5;
  return n;
}

int main(int argc, char **argv) {
  ekf_vio::init_logging(spdlog::level::info);
  auto log = ekf_vio::get_logger();

  if (argc < 2) {
    log->error("Usage: {} <euroc_sequence_path>", argv[0]);
    return 1;
  }

  // ------------------------------------------------------------------
  // Load dataset
  // ------------------------------------------------------------------
  EurocReader reader(argv[1]);
  if (!reader.load()) {
    log->error("Failed to load EuRoC sequence from: {}", argv[1]);
    return 1;
  }

  // ------------------------------------------------------------------
  // Stereo rectification (EuRoC images are raw / distorted)
  // ------------------------------------------------------------------
  StereoRectifier rectifier;
  rectifier.init();

  // ------------------------------------------------------------------
  // Initialise EKF + tracker (use rectified intrinsics)
  // ------------------------------------------------------------------
  auto cam = makeEurocCamera(rectifier);
  auto noise = makeEurocNoise();

  EKF ekf(cam, noise);

  StereoTracker::Params tp;
  tp.max_features = 300;
  tp.fast_threshold = 20;
  StereoTracker tracker(cam, tp);

  StereoVO::Params vp;
  vp.min_pnp_points = 8;
  vp.kf_tracked_ratio = 0.60;
  vp.kf_min_tracked = 80;
  vp.max_translation_m = 1.0;
  vp.max_rotation_deg = 20.0;
  StereoVO vo(cam, vp);

  // Offset to convert VO camera poses to EKF body poses
  // T_{ekfW ← bk} = T_offset * T_vo(k) * T_cam_imu
  // where T_offset = T_{ekfW ← b0} * T_{b0 ← c0} = T_wb_0 * T_cam_imu^{-1}
  Eigen::Isometry3d T_offset = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_cam_body = cam.T_cam_imu.inverse();
  bool offset_ready = false;

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
            ekf.state().q   = gt.q;
            ekf.state().p   = gt.p;
            ekf.state().v   = gt.v;
            ekf.state().b_g = gt.b_g;
            ekf.state().b_a = gt.b_a;
            log->info("Using ground truth at t={:.6f}s", stereo.timestamp);
          } else {
            ekf.state().q = Eigen::Quaterniond::Identity();
            ekf.state().p = Eigen::Vector3d::Zero();
            ekf.state().v = Eigen::Vector3d::Zero();
            log->info("No ground truth; starting at identity");
          }

          // Set realistic initial covariance (GT is good but not perfect)
          ekf.state().P.setZero();
          ekf.state().P.block<3,3>(0,0)  = Eigen::Matrix3d::Identity() * 1e-2;   // position ~10cm
          ekf.state().P.block<3,3>(3,3)  = Eigen::Matrix3d::Identity() * 1e-2;   // velocity ~0.1m/s
          ekf.state().P.block<3,3>(6,6)  = Eigen::Matrix3d::Identity() * 1e-3;   // orientation ~1.8deg
          ekf.state().P.block<3,3>(9,9)  = Eigen::Matrix3d::Identity() * 1e-4;   // gyro bias
          ekf.state().P.block<3,3>(12,12)= Eigen::Matrix3d::Identity() * 1e-2;   // accel bias
          initialized = true;
          last_imu_time = stereo.timestamp;
          return; // skip tracking on first frame
        }

        // Rectify raw images before tracking
        cv::Mat rect_left, rect_right;
        rectifier.rectify(stereo.left, stereo.right, rect_left, rect_right);

        const auto features = tracker.track(rect_left, rect_right);

        // Feature-based EKF update
        if (!features.empty()) {
          ekf.update(features);
        }

        // VO-based pose update (loosely coupled)
        Eigen::Isometry3d T_wc_vo = vo.process(features);
        if (!offset_ready) {
          // First VO frame: compute offset from GT-initialised EKF state
          Eigen::Isometry3d T_wb0 = Eigen::Isometry3d::Identity();
          T_wb0.linear() = ekf.state().q.toRotationMatrix();
          T_wb0.translation() = ekf.state().p;
          T_offset = T_wb0 * cam.T_cam_imu.inverse();
          offset_ready = true;
        }
        // Convert VO camera pose to body pose in EKF world frame
        Eigen::Isometry3d T_wb_vo = T_offset * T_wc_vo * cam.T_cam_imu;
        ekf.updateFromPose(
            T_wb_vo.translation(),
            Eigen::Quaterniond(T_wb_vo.rotation()),
            0.10,  // position noise (metres) — VO is ~10cm accurate
            0.03); // orientation noise (radians) — VO is ~1.7° accurate

        ++stereo_count;

        // Write trajectory
        const State &s = ekf.state();
        traj_out << std::fixed << std::setprecision(9) << stereo.timestamp
                 << "," << s.p.x() << "," << s.p.y() << "," << s.p.z() << ","
                 << s.q.w() << "," << s.q.x() << "," << s.q.y() << ","
                 << s.q.z() << "\n";

        // Print progress every 100 frames
        if (stereo_count % 100 == 0) {
          log->info("Frame {:4d}  t={:.3f}  feat={}  pos=({:.3f}, {:.3f}, {:.3f})",
                    stereo_count, stereo.timestamp, features.size(),
                    s.p.x(), s.p.y(), s.p.z());
        }
      });

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------
  const State &s = ekf.state();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames, {} IMU samples",
            stereo_count, reader.numImu());
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", s.p.x(), s.p.y(), s.p.z());
  log->info("Trajectory saved to euroc_traj.csv");

  return 0;
}
