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

option(COVERAGE "Enable gcov code coverage" OFF)
if(COVERAGE)
  add_compile_options(--coverage -fno-inline)
  add_link_options(--coverage)
endif()

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
install(TARGETS ekf_vio_core vio_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_math_utils test/test_math_utils.cpp)
  target_link_libraries(test_math_utils ekf_vio_core)

  ament_add_gtest(test_ekf test/test_ekf.cpp)
  target_link_libraries(test_ekf ekf_vio_core)
endif()

ament_package()
