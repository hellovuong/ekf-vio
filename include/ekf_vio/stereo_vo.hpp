/**
 * Keyframe-based Stereo Visual Odometry using PnP + RANSAC.
 *
 * Pipeline:
 *   1. StereoTracker provides features with persistent IDs + stereo depth
 *   2. Match tracked features against keyframe's 3D map points (by ID)
 *   3. Solve current pose via cv::solvePnPRansac
 *   4. Create new keyframe when tracking quality degrades
 */

#pragma once

#include "ekf_vio/types.hpp"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

namespace ekf_vio {

class StereoVO {
public:
  struct Params {
    int min_pnp_points = 10;            // minimum 3D-2D matches for PnP
    double pnp_reproj_thresh = 3.0;     // RANSAC reprojection threshold (px)
    double kf_tracked_ratio = 0.40;     // new KF when tracked/original < this
    int kf_min_tracked = 30;            // new KF when tracked count < this
    double kf_min_parallax_px = 30.0;   // new KF when median parallax > this
    double max_translation_m = 2.0;     // reject PnP if translation > this (per frame)
    double max_rotation_deg = 30.0;     // reject PnP if rotation > this (per frame)
  };

  explicit StereoVO(const StereoCamera &cam);
  StereoVO(const StereoCamera &cam, const Params &params);

  // -----------------------------------------------------------------------
  // Process one set of stereo features (output of StereoTracker::track).
  // Returns the current camera-frame pose in world coordinates: T_{world←cam}
  // -----------------------------------------------------------------------
  Eigen::Isometry3d process(const std::vector<Feature> &features);

  // Current pose accessor
  const Eigen::Isometry3d &pose() const { return T_wc_; }

  // Set the initial world pose (e.g. from ground truth)
  void setInitialPose(const Eigen::Isometry3d &T_wc) { T_wc_ = T_wc; }

  // Statistics
  int numKeyframeLandmarks() const;
  int numInliers() const { return last_inlier_count_; }

private:
  // A keyframe stores its world pose and 3D landmarks in its camera frame
  struct Keyframe {
    Eigen::Isometry3d T_wc = Eigen::Isometry3d::Identity();
    // feature_id → 3D point in this keyframe's camera frame
    std::unordered_map<int, Eigen::Vector3d> landmarks;
  };

  // Solve pose of current frame relative to keyframe using PnP
  // Returns true if successful; fills T_ck (T_{curr_cam ← kf_cam})
  bool solvePose(const std::vector<Eigen::Vector3d> &pts_3d,
                 const std::vector<cv::Point2f> &pts_2d,
                 Eigen::Isometry3d &T_ck, int &inlier_count);

  // Solve motion using 3D-3D alignment (SVD + RANSAC)
  // Returns true if successful; fills T_kf_curr (T_{kf_cam ← curr_cam})
  bool solveMotion3D3D(const std::vector<Eigen::Vector3d> &pts_kf,
                       const std::vector<Eigen::Vector3d> &pts_curr,
                       Eigen::Isometry3d &T_kf_curr, int &inlier_count);

  // Decide whether to spawn a new keyframe
  bool shouldCreateKeyframe(int tracked_from_kf, int total_in_kf,
                            const std::vector<Feature> &features) const;

  // Create keyframe from current features at current pose
  void createKeyframe(const std::vector<Feature> &features);

  StereoCamera cam_;
  Params params_;

  Eigen::Isometry3d T_wc_ = Eigen::Isometry3d::Identity(); // current pose
  Keyframe keyframe_;
  bool has_keyframe_ = false;
  int last_inlier_count_ = 0;
};

} // namespace ekf_vio
