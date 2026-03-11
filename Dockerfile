# ═══════════════════════════════════════════════════════════════
# Stage: deps — ROS 2 Jazzy + all build dependencies
#
# Shared by build/test (CI) and dev (local) stages.
# ═══════════════════════════════════════════════════════════════
FROM ros:jazzy-ros-base AS deps

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ccache \
    ninja-build \
    mold \
    git \
    lcov \
    python3-colcon-common-extensions \
    libeigen3-dev \
    libopencv-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    ros-jazzy-rclcpp \
    ros-jazzy-sensor-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-cv-bridge \
    ros-jazzy-message-filters \
    ros-jazzy-ament-cmake-gtest \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y wget lsb-release software-properties-common gnupg && \
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 19 all

RUN update-alternatives    --install /usr/bin/clang        clang        /usr/bin/clang-19        100 \
    && update-alternatives --install /usr/bin/clang++      clang++      /usr/bin/clang++-19      100 \
    && update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-19   100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100 \
    && update-alternatives --install /usr/bin/llvm-cov     llvm-cov     /usr/bin/llvm-cov-19     100 \
    && update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-19 100 \
    && printf '#!/bin/bash\nexec /usr/bin/llvm-cov-19 gcov "$@"\n' > /usr/local/bin/llvm-cov-gcov \
    && chmod +x /usr/local/bin/llvm-cov-gcov

# Sophus (header-only, no apt package on Ubuntu 24.04)
RUN git clone --depth 1 --branch 1.22.10 https://github.com/strasdat/Sophus.git /tmp/sophus && \
    cmake -S /tmp/sophus -B /tmp/sophus/build \
      -DCMAKE_BUILD_TYPE=Release -DSOPHUS_USE_BASIC_LOGGING=ON && \
    cmake --install /tmp/sophus/build && \
    rm -rf /tmp/sophus

ENV CC=clang \
    CXX=clang++ \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G \
    CMAKE_GENERATOR=Ninja \
    CMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" \
    CMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold"

RUN mkdir -p /ccache && ccache --max-size=5G

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# ═══════════════════════════════════════════════════════════════
# Stage: build — compile the project
# ═══════════════════════════════════════════════════════════════
FROM deps AS build

WORKDIR /ws
COPY . /ws/src/ekf-vio

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --event-handlers compile_commands+ console_cohesion+ \
      --packages-select ekf_vio \
      --cmake-args \
        -GNinja \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DBUILD_TESTING=ON \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DUSE_COVERAGE=ON

# ═══════════════════════════════════════════════════════════════
# Stage: test — run unit tests (CI stops here)
#
#   docker build --target test .
# ═══════════════════════════════════════════════════════════════
FROM build AS test

RUN source /opt/ros/jazzy/setup.bash && \
    source /ws/install/setup.bash && \
    colcon test \
      --packages-select ekf_vio \
      --return-code-on-test-failure && \
    colcon test-result --verbose

# Generate and Filter Coverage
RUN lcov --capture --config-file /ws/src/ekf-vio/.lcovrc --directory build/ekf_vio --output-file coverage_raw.info \
    --ignore-errors mismatch,gcov,unused,inconsistent \
    --gcov-tool /usr/local/bin/llvm-cov-gcov && \
    lcov --extract coverage_raw.info "/ws/src/ekf-vio/*" --output-file coverage_cleaned.info && \
    lcov --remove coverage_cleaned.info "*/test/*" --output-file /ws/coverage.info

# ═══════════════════════════════════════════════════════════════
# Stage: viz — build with Rerun visualization support
#
#   docker build --target viz -t ekf-vio-viz .
#
# Linux (live viewer — viewer runs on the host):
#   Host:      rerun
#   Container: docker run --rm -it --network host \
#                -v /path/to/MH_01_easy:/data \
#                ekf-vio-viz \
#                bash -c "source /opt/ros/jazzy/setup.bash && \
#                         source /ws/install/setup.bash && \
#                         /ws/install/lib/ekf_vio/euroc_rerun_runner \
#                           /data config/euroc.yaml"
#
# macOS / Windows (file-based — save .rrd, open on host):
#   Container: docker run --rm -it \
#                -v /path/to/MH_01_easy:/data \
#                -v $(pwd)/results:/results \
#                ekf-vio-viz \
#                bash -c "source /opt/ros/jazzy/setup.bash && \
#                         source /ws/install/setup.bash && \
#                         /ws/install/lib/ekf_vio/euroc_rerun_runner \
#                           /data config/euroc.yaml --save /results/ekf_vio.rrd"
#   Host:      rerun results/ekf_vio.rrd
# ═══════════════════════════════════════════════════════════════
FROM deps AS viz

WORKDIR /ws
COPY . /ws/src/ekf-vio

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --event-handlers compile_commands+ console_cohesion+ \
      --packages-select ekf_vio \
      --cmake-args \
        -GNinja \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DWITH_RERUN=ON

CMD ["bash"]

# ═══════════════════════════════════════════════════════════════
# Stage: dev — local development environment (default target)
#
#   docker build -t ekf-vio-dev .
# ═══════════════════════════════════════════════════════════════
FROM deps AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ripgrep \
    python3-pip \
    python3-vcstool \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Neovim (ARM — for Apple Silicon dev containers)
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-arm64.tar.gz && \
    rm -rf /opt/nvim-linux-arm64 && \
    tar -C /opt -xzf nvim-linux-arm64.tar.gz && \
    rm nvim-linux-arm64.tar.gz
RUN git clone https://github.com/hellovuong/astronvim_config.git ~/.config/nvim
RUN echo 'export PATH="$PATH:/opt/nvim-linux-arm64/bin"' >> /root/.bashrc

WORKDIR /workspace
RUN mkdir -p /workspace/src/ekf-vio
COPY . /workspace/src/ekf-vio

CMD ["bash"]
