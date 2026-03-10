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
# Warning flags (applied per-target below to avoid leaking into dependencies)
# ──────────────────────────────────────────────────────────────────────────────
set(EKF_VIO_WARN_FLAGS
    -Wall -Wextra -Wpedantic -Werror
    -Wno-unknown-warning-option   # lets clang/clang-tidy ignore GCC-specific flags
)
# GCC 13+ false-positives in fmt/spdlog templates
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND EKF_VIO_WARN_FLAGS -Wno-error=stringop-overflow -Wno-error=array-bounds)
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
find_package(Sophus REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# ──────────────────────────────────────────────────────────────────────────────
# Library: ekf_vio_core  (EKF + StereoTracker)
# ──────────────────────────────────────────────────────────────────────────────
add_library(ekf_vio_core
    src/ekf.cpp
    src/stereo_tracker.cpp
    src/stereo_vo.cpp
    src/stereo_rectifier.cpp
    src/euroc_reader.cpp
)

target_include_directories(ekf_vio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ekf_vio_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Sophus::Sophus
    spdlog::spdlog
    yaml-cpp
)
target_compile_options(ekf_vio_core PRIVATE ${EKF_VIO_WARN_FLAGS})
set_target_properties(ekf_vio_core PROPERTIES CXX_CLANG_TIDY "clang-tidy;--warnings-as-errors=*")

# ──────────────────────────────────────────────────────────────────────────────
# Executable: euroc_runner  (standalone, no ROS2 dependency)
# ──────────────────────────────────────────────────────────────────────────────
add_executable(euroc_runner app/euroc_main.cpp)
target_link_libraries(euroc_runner PRIVATE ekf_vio_core)
target_compile_options(euroc_runner PRIVATE ${EKF_VIO_WARN_FLAGS})
set_target_properties(euroc_runner PROPERTIES CXX_CLANG_TIDY "clang-tidy;--warnings-as-errors=*")

add_executable(euroc_vo_runner app/euroc_vo_main.cpp)
target_link_libraries(euroc_vo_runner PRIVATE ekf_vio_core)
target_compile_options(euroc_vo_runner PRIVATE ${EKF_VIO_WARN_FLAGS})
set_target_properties(euroc_vo_runner PROPERTIES CXX_CLANG_TIDY "clang-tidy;--warnings-as-errors=*")

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
target_compile_options(vio_node PRIVATE ${EKF_VIO_WARN_FLAGS})
set_target_properties(vio_node PROPERTIES CXX_CLANG_TIDY "clang-tidy;--warnings-as-errors=*")

# ──────────────────────────────────────────────────────────────────────────────
# Optional: Rerun visualization runner
#
# Build with:
#   colcon build --cmake-args -DWITH_RERUN=ON
# ──────────────────────────────────────────────────────────────────────────────
option(WITH_RERUN "Build Rerun visualization runner (euroc_rerun_runner)" OFF)

if(WITH_RERUN)
  include(FetchContent)
  FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.19.0/rerun_cpp_sdk.zip
  )
  FetchContent_MakeAvailable(rerun_sdk)
  set_target_properties(rerun_sdk PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:rerun_sdk,INTERFACE_INCLUDE_DIRECTORIES>)

  add_executable(euroc_rerun_runner app/euroc_rerun_main.cpp)
  target_link_libraries(euroc_rerun_runner PRIVATE ekf_vio_core rerun_sdk)
  target_compile_options(euroc_rerun_runner PRIVATE ${EKF_VIO_WARN_FLAGS})
  set_target_properties(euroc_rerun_runner PROPERTIES CXX_CLANG_TIDY "clang-tidy;--warnings-as-errors=*;-checks=-performance-move-const-arg")

  install(TARGETS euroc_rerun_runner
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

# ──────────────────────────────────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────────────────────────────────
install(TARGETS ekf_vio_core vio_node euroc_runner euroc_vo_runner
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

  ament_add_gtest(stereo_tracker_test test/stereo_tracker_test.cpp)
  target_link_libraries(stereo_tracker_test ekf_vio_core)
  target_compile_options(stereo_tracker_test PRIVATE ${EKF_VIO_WARN_FLAGS})

  ament_add_gtest(stereo_vo_test test/stereo_vo_test.cpp)
  target_link_libraries(stereo_vo_test ekf_vio_core)
  target_compile_options(stereo_vo_test PRIVATE ${EKF_VIO_WARN_FLAGS})

  ament_add_gtest(ekf_test test/ekf_test.cpp)
  target_link_libraries(ekf_test ekf_vio_core)
  target_compile_options(ekf_test PRIVATE ${EKF_VIO_WARN_FLAGS})
endif()

ament_package()
