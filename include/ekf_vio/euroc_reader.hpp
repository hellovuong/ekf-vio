// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ekf_vio/types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace ekf_vio {

// ============================================================================
//  EurocReader — reads EuRoC MAV dataset directly from disk
//
//  Expected directory layout:
//    <sequence_path>/mav0/cam0/data.csv   + data/*.png
//    <sequence_path>/mav0/cam1/data.csv   + data/*.png
//    <sequence_path>/mav0/imu0/data.csv
//    <sequence_path>/mav0/state_groundtruth_estimate0/data.csv  (optional)
// ============================================================================

struct StereoImages {
  double timestamp;  // seconds
  cv::Mat left;
  cv::Mat right;
};

struct GroundTruth {
  double timestamp;      // seconds
  Eigen::Vector3d p;     // position
  Eigen::Quaterniond q;  // orientation (w, x, y, z)
  Eigen::Vector3d v;     // velocity
  Eigen::Vector3d b_g;   // gyro bias
  Eigen::Vector3d b_a;   // accel bias
};

// A single chronological event: either an IMU sample or a stereo frame
struct DataEvent {
  enum Type : std::uint8_t { IMU, STEREO };
  Type type;
  // Index into the respective array (imu_data or stereo_timestamps)
  size_t index;
};

class EurocReader {
 public:
  // Construct with path to a sequence root, e.g. ".../MH_01_easy"
  explicit EurocReader(const std::string& sequence_path);

  // Load all CSV metadata (and optionally ground truth).
  // Images are NOT loaded here — only timestamps and paths.
  bool load();

  // Total counts
  [[nodiscard]] size_t numImu() const { return imu_data_.size(); }
  [[nodiscard]] size_t numStereo() const { return stereo_timestamps_.size(); }
  [[nodiscard]] size_t numEvents() const { return events_.size(); }

  // Access pre-parsed IMU data
  [[nodiscard]] const std::vector<ImuData>& imuData() const { return imu_data_; }

  // Access merged timeline
  [[nodiscard]] const std::vector<DataEvent>& events() const { return events_; }

  // Load stereo images for a given stereo index (lazy, from disk)
  [[nodiscard]] StereoImages loadStereo(size_t stereo_index) const;

  // Access ground truth (empty if not available)
  [[nodiscard]] const std::vector<GroundTruth>& groundTruth() const { return ground_truth_; }

  // Find the closest ground truth entry to a timestamp (seconds)
  bool closestGroundTruth(double t, GroundTruth& out) const;

  // Convenience: iterate through the dataset in chronological order.
  // Callbacks are invoked for each event type.
  using ImuCallback = std::function<void(const ImuData&)>;
  using StereoCallback = std::function<void(const StereoImages&)>;
  void replay(const ImuCallback& on_imu, const StereoCallback& on_stereo) const;

 private:
  bool loadImu();
  bool loadCameraTimestamps(const std::string& cam_dir,
                            std::vector<std::pair<double, std::string>>& out);
  bool loadGroundTruth();
  void buildTimeline();

  std::string base_path_;  // .../MH_01_easy/mav0

  std::vector<ImuData> imu_data_;

  // cam0 / cam1 timestamps + image filenames
  std::vector<std::pair<double, std::string>> cam0_entries_;
  std::vector<std::pair<double, std::string>> cam1_entries_;

  // Merged stereo timestamps (using cam0 as reference)
  std::vector<double> stereo_timestamps_;

  // Chronologically sorted event list
  std::vector<DataEvent> events_;

  // Optional ground truth
  std::vector<GroundTruth> ground_truth_;
};

}  // namespace ekf_vio
