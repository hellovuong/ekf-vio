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

# Sophus (header-only, no apt package on Ubuntu 24.04)
RUN git clone --depth 1 --branch 1.22.10 https://github.com/strasdat/Sophus.git /tmp/sophus && \
    cmake -S /tmp/sophus -B /tmp/sophus/build \
      -DCMAKE_BUILD_TYPE=Release -DSOPHUS_USE_BASIC_LOGGING=ON && \
    cmake --install /tmp/sophus/build && \
    rm -rf /tmp/sophus

ENV CC="ccache gcc" \
    CXX="ccache g++" \
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
COPY . /ws/src/ekf_vio

RUN apt-get update && apt-get install -y --no-install-recommends clang-tidy \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --packages-select ekf_vio \
      --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
        -DCMAKE_CXX_CLANG_TIDY="clang-tidy;--warnings-as-errors=*"

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

# ═══════════════════════════════════════════════════════════════
# Stage: dev — local development environment (default target)
#
#   docker build -t ekf-vio-dev .
# ═══════════════════════════════════════════════════════════════
FROM deps AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
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
