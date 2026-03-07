cmake_minimum_required(VERSION 3.14)
project(ekf_vio)

# ──────────────────────────────────────────────────────────────────────────────
# Build type
# ──────────────────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED
    COMPONENTS core imgproc features2d video calib3d)

# ──────────────────────────────────────────────────────────────────────────────
# Library: ekf_vio_core  (EKF + StereoTracker)
# ──────────────────────────────────────────────────────────────────────────────
add_library(ekf_vio_core
    src/ekf.cpp
    src/stereo_tracker.cpp
    src/euroc_reader.cpp
)

target_include_directories(ekf_vio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ekf_vio_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

# ──────────────────────────────────────────────────────────────────────────────
# Executable: euroc_runner  (standalone, no ROS2 dependency)
# ──────────────────────────────────────────────────────────────────────────────
add_executable(euroc_runner app/euroc_main.cpp)
target_link_libraries(euroc_runner PRIVATE ekf_vio_core)

# ──────────────────────────────────────────────────────────────────────────────
# Executable: vio_node
# ──────────────────────────────────────────────────────────────────────────────
add_executable(vio_node app/vio_node.cpp)

target_link_libraries(vio_node PRIVATE ekf_vio_core
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${nav_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${tf2_ros_TARGETS}
    ${cv_bridge_TARGETS}
    ${message_filters_TARGETS}
)

# ──────────────────────────────────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────────────────────────────────
install(TARGETS ekf_vio_core vio_node euroc_runner
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
