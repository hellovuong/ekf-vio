// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/ekf.hpp"
#include "ekf_vio/stereo_tracker.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <functional>

namespace ekf_vio {

class VIONode : public rclcpp::Node {
 public:
  VIONode() : Node("ekf_vio_node") {
    // ----------------------------------------------------------------
    // Declare and load parameters
    // ----------------------------------------------------------------
    this->declare_parameter("camera.fx", 458.654);
    this->declare_parameter("camera.fy", 457.296);
    this->declare_parameter("camera.cx", 367.215);
    this->declare_parameter("camera.cy", 248.375);
    this->declare_parameter("camera.baseline", 0.110);

    // IMU-to-camera extrinsics as flat 4x4 row-major
    this->declare_parameter("T_cam_imu",
                            std::vector<double>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});

    // Process noise
    this->declare_parameter("noise.sigma_gyro", 1.6968e-4);
    this->declare_parameter("noise.sigma_accel", 2.0000e-3);
    this->declare_parameter("noise.sigma_gyro_bias", 1.9393e-5);
    this->declare_parameter("noise.sigma_accel_bias", 3.0000e-5);
    this->declare_parameter("noise.sigma_pixel", 1.5);

    // ----------------------------------------------------------------
    // Build camera & noise configs
    // ----------------------------------------------------------------
    StereoCamera cam;
    cam.fx = this->get_parameter("camera.fx").as_double();
    cam.fy = this->get_parameter("camera.fy").as_double();
    cam.cx = this->get_parameter("camera.cx").as_double();
    cam.cy = this->get_parameter("camera.cy").as_double();
    cam.baseline = this->get_parameter("camera.baseline").as_double();

    auto T_flat = this->get_parameter("T_cam_imu").as_double_array();
    Eigen::Matrix4d T_mat;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        T_mat(r, c) = T_flat[(r * 4) + c];
      }
    }
    cam.T_cam_imu = Sophus::SE3d(T_mat);

    EKF::NoiseParams noise;
    noise.sigma_gyro = this->get_parameter("noise.sigma_gyro").as_double();
    noise.sigma_accel = this->get_parameter("noise.sigma_accel").as_double();
    noise.sigma_gyro_bias = this->get_parameter("noise.sigma_gyro_bias").as_double();
    noise.sigma_accel_bias = this->get_parameter("noise.sigma_accel_bias").as_double();
    noise.sigma_pixel = this->get_parameter("noise.sigma_pixel").as_double();

    // ----------------------------------------------------------------
    // Initialise EKF and tracker
    // ----------------------------------------------------------------
    ekf_ = std::make_unique<EKF>(cam, noise);
    tracker_ = std::make_unique<StereoTracker>(cam, StereoTracker::Params());

    // ----------------------------------------------------------------
    // Subscribers
    // ----------------------------------------------------------------
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Imu::ConstSharedPtr& msg) { imuCallback(msg); });

    // Approximate-time sync for stereo pair
    left_sub_.subscribe(this, "/camera/left/image_raw");
    right_sub_.subscribe(this, "/camera/right/image_raw");

    using StereoPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                                         sensor_msgs::msg::Image>;
    stereo_sync_ = std::make_shared<message_filters::Synchronizer<StereoPolicy>>(
        StereoPolicy(10), left_sub_, right_sub_);
    // clang-format off
    stereo_sync_->registerCallback(
        std::bind(&VIONode::stereoCallback, this,                    // NOLINT(modernize-avoid-bind)
                  std::placeholders::_1, std::placeholders::_2));
    // clang-format on

    // ----------------------------------------------------------------
    // Publishers
    // ----------------------------------------------------------------
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/vio/odometry", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(this->get_logger(), "EKF-VIO node started. Waiting for data...");
  }

 private:
  // ----------------------------------------------------------------
  // IMU callback — runs the EKF predict step
  // ----------------------------------------------------------------
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    const double t = msg->header.stamp.sec + (msg->header.stamp.nanosec * 1e-9);

    ImuData imu;
    imu.timestamp = t;
    imu.gyro = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
    imu.accel = {msg->linear_acceleration.x, msg->linear_acceleration.y,
                 msg->linear_acceleration.z};

    if (!initialized_) {
      // Skip until first stereo frame initialises attitude
      last_imu_time_ = t;
      return;
    }

    const double dt = t - last_imu_time_;
    if (dt <= 0.0 || dt > 0.5) {
      last_imu_time_ = t;
      return;
    }

    ekf_->predict(imu, dt);
    last_imu_time_ = t;

    publishOdometry(msg->header.stamp);
  }

  // ----------------------------------------------------------------
  // Stereo callback — runs tracker + EKF update
  // ----------------------------------------------------------------
  void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
    cv::Mat left;
    cv::Mat right;
    try {
      left = cv_bridge::toCvShare(left_msg, "mono8")->image;
      right = cv_bridge::toCvShare(right_msg, "mono8")->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    if (!initialized_) {
      // On first frame: align gravity with accelerometer reading
      // (Assumes the robot starts roughly stationary)
      initAttitude();
      initialized_ = true;
      last_imu_time_ = left_msg->header.stamp.sec + left_msg->header.stamp.nanosec * 1e-9;
      RCLCPP_INFO(this->get_logger(), "EKF-VIO initialised.");
    }

    const auto features = tracker_->track(left, right);

    if (features.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No features tracked.");
      return;
    }

    ekf_->update(features);
    RCLCPP_DEBUG(this->get_logger(), "Updated with %zu features.", features.size());
  }

  // ----------------------------------------------------------------
  // Attitude initialisation from static accelerometer reading
  // (assumes gravity-aligned start)
  // ----------------------------------------------------------------
  void initAttitude() {
    // Default: assume camera starts level, z-axis up
    // In practice: average first N IMU readings and align gravity
    ekf_->state().T_wb = Sophus::SE3d();
    ekf_->state().v = Eigen::Vector3d::Zero();
  }

  // ----------------------------------------------------------------
  // Publish nav_msgs/Odometry + TF
  // ----------------------------------------------------------------
  void publishOdometry(const rclcpp::Time& stamp) {
    const State& s = ekf_->state();
    const auto& pos = s.T_wb.translation();
    const auto& quat = s.T_wb.unit_quaternion();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "imu";

    odom.pose.pose.position.x = pos.x();
    odom.pose.pose.position.y = pos.y();
    odom.pose.pose.position.z = pos.z();
    odom.pose.pose.orientation.x = quat.x();
    odom.pose.pose.orientation.y = quat.y();
    odom.pose.pose.orientation.z = quat.z();
    odom.pose.pose.orientation.w = quat.w();

    odom.twist.twist.linear.x = s.v.x();
    odom.twist.twist.linear.y = s.v.y();
    odom.twist.twist.linear.z = s.v.z();

    // Copy pose covariance (top-left 6×6 of 15×15 error-state cov)
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        odom.pose.covariance[(r * 6) + c] = s.P(r, c);
      }
    }

    odom_pub_->publish(odom);

    // TF
    geometry_msgs::msg::TransformStamped tf;
    tf.header = odom.header;
    tf.child_frame_id = "imu";
    tf.transform.translation.x = pos.x();
    tf.transform.translation.y = pos.y();
    tf.transform.translation.z = pos.z();
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  // ----------------------------------------------------------------
  std::unique_ptr<EKF> ekf_;
  std::unique_ptr<StereoTracker> tracker_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;

  using StereoPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                                       sensor_msgs::msg::Image>;
  std::shared_ptr<message_filters::Synchronizer<StereoPolicy>> stereo_sync_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  bool initialized_ = false;
  double last_imu_time_ = 0.0;
};

}  // namespace ekf_vio

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ekf_vio::VIONode>());
  rclcpp::shutdown();
  return 0;
}
