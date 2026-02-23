# ================================
# Base: ROS 2 Jazzy (Ubuntu 24.04)
# ================================
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ================================
# Install build tools & utilities
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    ninja-build \
    mold \
    git \
    curl \
    wget \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-pip \
    libeigen3-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Install ROS dependencies
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rclcpp \
    ros-jazzy-sensor-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-cv-bridge \
    ros-jazzy-message-filters \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Configure ccache
# ================================
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

RUN mkdir -p /ccache && ccache --max-size=5G
# ================================
# Enable mold linker + Ninja
# ================================
ENV CMAKE_GENERATOR=Ninja
ENV CMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold"
ENV CMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold"


# ================================
# Configure nvim 
# ================================
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-arm64.tar.gz && \
    rm -rf /opt/nvim-linux-x86_64 && \
    tar -C /opt -xzf nvim-linux-arm64.tar.gz
RUN git clone https://github.com/hellovuong/astronvim_config.git ~/.config/nvim


# ================================
# Workspace
# ================================
WORKDIR /workspace
RUN mkdir -p /workspace/src/ekf-vio
COPY ./ /workspace/src/ekf-vio

# Source ROS automatically
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "export PATH="$PATH:/opt/nvim-linux-arm64/bin"" >> /root/.bashrc

CMD ["bash"]
