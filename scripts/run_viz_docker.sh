#!/usr/bin/env bash
# run_viz_docker.sh — Run the Rerun-enabled EKF-VIO visualization in Docker
#
# Usage:
#   ./scripts/run_viz_docker.sh <euroc_sequence_path> [--save]
#
# Arguments:
#   <euroc_sequence_path>  Path to an EuRoC sequence folder (e.g. /data/MH_01_easy)
#   --save                 Save to .rrd file instead of live TCP connection
#                          (required on macOS / Windows)
#
# Prerequisites:
#   1. Build the viz image:
#        docker build --target viz -t ekf-vio-viz .
#   2. (Live mode) Install and start the Rerun viewer on the host:
#        pip install rerun-sdk && rerun

set -euo pipefail

SEQUENCE="${1:?Usage: $0 <euroc_sequence_path> [--save]}"
MODE="${2:-}"

IMAGE="ekf-vio-viz"
CONFIG="config/euroc.yaml"
RUNNER="/ws/install/lib/ekf_vio/euroc_rerun_runner"

# Entrypoint shared by both modes
ENTRYPOINT="source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash"

if [[ "$MODE" == "--save" ]]; then
  # ── File-based mode (macOS / Windows / CI) ───────────────────────────────
  # Saves ekf_vio.rrd into ./results/ on the host, then open with:
  #   rerun results/ekf_vio.rrd
  RESULTS_DIR="$(pwd)/results"
  mkdir -p "$RESULTS_DIR"

  echo "Running in SAVE mode  →  output: $RESULTS_DIR/ekf_vio.rrd"
  echo "After the run, open with:  rerun $RESULTS_DIR/ekf_vio.rrd"

  docker run --rm -it \
    -v "${SEQUENCE}:/data" \
    -v "${RESULTS_DIR}:/results" \
    "${IMAGE}" \
    bash -c "${ENTRYPOINT} && \
              ${RUNNER} /data ${CONFIG} --save /results/ekf_vio.rrd"
else
  # ── Live TCP mode (Linux --network host) ─────────────────────────────────
  # Requires the Rerun viewer to be running on the host BEFORE starting Docker:
  #   rerun
  if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: Live mode (--network host) only works on Linux."
    echo "       Use --save mode on macOS / Windows."
    exit 1
  fi

  echo "Running in LIVE mode  →  connecting to host Rerun viewer at 127.0.0.1:9876"
  echo "Make sure 'rerun' is running on the host before launching this script."

  docker run --rm -it \
    --network host \
    -v "${SEQUENCE}:/data" \
    "${IMAGE}" \
    bash -c "${ENTRYPOINT} && \
              ${RUNNER} /data ${CONFIG} --connect 127.0.0.1:9876"
fi
