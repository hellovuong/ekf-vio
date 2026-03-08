// ============================================================================
//  Standalone EuRoC dataset runner for Keyframe-based Stereo VO
//
//  Usage:  ./euroc_vo_runner <path_to_sequence>
//          e.g.  ./euroc_vo_runner /data/EuRoC/V1_01_easy
//
//  Pure visual odometry — no IMU.  Outputs trajectory CSV for evaluation.
// ============================================================================

#include "ekf_vio/euroc_reader.hpp"
#include "ekf_vio/logging.hpp"
#include "ekf_vio/stereo_tracker.hpp"
#include "ekf_vio/stereo_vo.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace ekf_vio;

// ============================================================================
// Stereo rectification for EuRoC raw images (same as EKF runner)
// ============================================================================
struct StereoRectifier {
  cv::Mat map1_left, map2_left;
  cv::Mat map1_right, map2_right;
  double fx, fy, cx, cy, baseline;

  void init() {
    cv::Mat K0 = (cv::Mat_<double>(3, 3) << 458.654, 0.0, 367.215, 0.0,
                  457.296, 248.375, 0.0, 0.0, 1.0);
    cv::Mat D0 =
        (cv::Mat_<double>(4, 1) << -0.28340811, 0.07395907, 0.00019359,
         1.76187114e-05);

    cv::Mat K1 = (cv::Mat_<double>(3, 3) << 457.587, 0.0, 379.999, 0.0,
                  456.134, 255.238, 0.0, 0.0, 1.0);
    cv::Mat D1 =
        (cv::Mat_<double>(4, 1) << -0.28368365, 0.07451284, -0.00010473,
         -3.55590700e-05);

    // T_BS0 and T_BS1 from EuRoC sensor.yaml.
    // EuRoC convention: T_BS = sensor → body.
    Eigen::Matrix4d T0, T1;
    T0 << 0.0148655429818, -0.999880929698, 0.00414029679422,
        -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948,
        -0.064676986768, -0.0257744366974, 0.00375618835797, 0.999660727178,
        0.00981073058949, 0.0, 0.0, 0.0, 1.0;
    T1 << 0.0125552670891, -0.999755099723, 0.0182237714554,
        -0.0198435579556, 0.999598781151, 0.0130119051815, 0.0251588363115,
        0.0453689425024, -0.0253898008918, 0.0179005838253, 0.999517347078,
        0.00786212447038, 0.0, 0.0, 0.0, 1.0;

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
    cv::stereoRectify(K0, D0, K1, D1, imgSize, R_cv, T_cv, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0, imgSize);

    cv::initUndistortRectifyMap(K0, D0, R1, P1, imgSize, CV_32FC1, map1_left,
                                map2_left);
    cv::initUndistortRectifyMap(K1, D1, R2, P2, imgSize, CV_32FC1, map1_right,
                                map2_right);

    double f_rect = P1.at<double>(0, 0);
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

// Build StereoCamera from rectified intrinsics
static StereoCamera makeCamera(const StereoRectifier &rect) {
  StereoCamera cam;
  cam.fx = rect.fx;
  cam.fy = rect.fy;
  cam.cx = rect.cx;
  cam.cy = rect.cy;
  cam.baseline = rect.baseline;
  cam.T_cam_imu = Eigen::Isometry3d::Identity(); // unused in VO mode
  return cam;
}

// ============================================================================
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
  // Stereo rectification
  // ------------------------------------------------------------------
  StereoRectifier rectifier;
  rectifier.init();

  // ------------------------------------------------------------------
  // Initialise tracker + VO
  // ------------------------------------------------------------------
  auto cam = makeCamera(rectifier);

  StereoTracker::Params tp;
  tp.max_features = 300;
  tp.fast_threshold = 20;
  StereoTracker tracker(cam, tp);

  StereoVO::Params vp;
  vp.min_pnp_points = 8;
  vp.pnp_reproj_thresh = 3.0;
  vp.kf_tracked_ratio = 0.60;
  vp.kf_min_tracked = 80;
  vp.max_translation_m = 1.0;
  vp.max_rotation_deg = 20.0;
  StereoVO vo(cam, vp);

  // Optionally initialise from GT (for fair comparison with VIO methods)
  GroundTruth gt0;
  bool have_gt = false;

  // Trajectory output
  std::ofstream traj_out("euroc_traj.csv");
  traj_out << "# timestamp,px,py,pz,qw,qx,qy,qz\n";

  int frame_count = 0;

  // ------------------------------------------------------------------
  // Replay — only process stereo frames (ignore IMU)
  // ------------------------------------------------------------------
  reader.replay(
      // IMU callback (no-op)
      [](const ImuData &) {},

      // Stereo callback
      [&](const StereoImages &stereo) {
        if (stereo.left.empty() || stereo.right.empty())
          return;

        // Initialise world pose from GT on first frame
        if (frame_count == 0 && !have_gt) {
          GroundTruth gt;
          if (reader.closestGroundTruth(stereo.timestamp, gt)) {
            // GT gives T_{world ← body}. For VO we work in camera frame.
            // Since we don't use IMU, we define world = first camera frame.
            // But for evaluation alignment, starting from GT is fine.
            Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
            T_wb.linear() = gt.q.toRotationMatrix();
            T_wb.translation() = gt.p;
            // We output body-frame poses for comparison with GT
            // VO estimates camera poses; we'll just output camera poses
            // and let the evaluation script do Umeyama alignment.
            // Start from identity — evaluation aligns automatically.
          }
          have_gt = true;
        }

        // Rectify
        cv::Mat rect_left, rect_right;
        rectifier.rectify(stereo.left, stereo.right, rect_left, rect_right);

        // Track features
        const auto features = tracker.track(rect_left, rect_right);

        // Estimate pose
        Eigen::Isometry3d T_wc = vo.process(features);

        ++frame_count;

        // Extract position and orientation for output
        const Eigen::Vector3d p = T_wc.translation();
        const Eigen::Quaterniond q(T_wc.rotation());

        traj_out << std::fixed << std::setprecision(9) << stereo.timestamp
                 << "," << p.x() << "," << p.y() << "," << p.z() << ","
                 << q.w() << "," << q.x() << "," << q.y() << "," << q.z()
                 << "\n";

        // Progress
        if (frame_count % 100 == 0) {
          log->info("Frame {:4d}  t={:.3f}  feat={}  inliers={}  kf_lm={}  pos=({:.3f}, {:.3f}, {:.3f})",
                    frame_count, stereo.timestamp, features.size(),
                    vo.numInliers(), vo.numKeyframeLandmarks(),
                    p.x(), p.y(), p.z());
        }
      });

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------
  const Eigen::Vector3d p = vo.pose().translation();
  log->info("=== Done ===");
  log->info("Processed {} stereo frames", frame_count);
  log->info("Final pos: ({:.3f}, {:.3f}, {:.3f})", p.x(), p.y(), p.z());
  log->info("Trajectory saved to euroc_traj.csv");

  return 0;
}
