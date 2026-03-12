#!/usr/bin/env bash
# Copyright (c) 2026, Long Vuong
# SPDX-License-Identifier: BSD-3-Clause
#
# Download EuRoC MAV test sequences for regression testing.
# Data is cached in ~/.cache/ekf-vio-data to avoid re-downloading on repeated runs.
#
# Usage:
#   ./scripts/download_testdata.sh [sequence ...]
#
# Examples:
#   ./scripts/download_testdata.sh                         # default: V1_01_easy
#   ./scripts/download_testdata.sh V1_01_easy V1_02_medium
#
# The script prefers trimmed 60-second subsets (~150 MB, hosted on this
# repo's GitHub Releases as v0-testdata) for CI speed.
# Set USE_ASL=1 to download the full sequences from the ETH Research Collection
# (https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f).
# Note: the ETH collection serves grouped ZIPs (~5–12 GB per environment).
#
# Output layout in cache dir:
#   ~/.cache/ekf-vio-data/
#     V1_01_easy/
#       mav0/
#         cam0/data/          ← left images
#         cam1/data/          ← right images
#         imu0/data.csv
#         state_groundtruth_estimate0/data.csv

set -euo pipefail

CACHE_DIR="${EKF_VIO_DATA_DIR:-$HOME/.cache/ekf-vio-data}"
USE_ASL="${USE_ASL:-0}"

# GitHub Release hosting trimmed 60-second subsets (~150 MB each, zstd-compressed)
# Update this tag when test data is re-published.
GH_RELEASE_BASE="https://github.com/hellovuong/ekf-vio/releases/download/v0-testdata"

# Official ETH Research Collection (grouped ZIPs — one ZIP per environment).
# Source: https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f
# Each ZIP contains all sequences for that environment; only the requested
# sequence folder is extracted.  Group ZIPs are cached to avoid re-downloading
# when multiple sequences from the same environment are requested.
#   Vicon Room 1 (~5.7 GB): V1_01_easy … V1_03_difficult
#   Vicon Room 2 (~5.7 GB): V2_01_easy … V2_03_difficult
declare -A ETH_GROUP_URLS=(
  [vicon_room1]="https://www.research-collection.ethz.ch/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/download"
  [vicon_room2]="https://www.research-collection.ethz.ch/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/download"
)

seq_to_group() {
  case "$1" in
    V1_*) echo "vicon_room1" ;;
    V2_*) echo "vicon_room2" ;;
    *) echo "" ;;
  esac
}

DEFAULT_SEQUENCES=(V1_01_easy)
SEQUENCES=("${@:-${DEFAULT_SEQUENCES[@]}}")

mkdir -p "$CACHE_DIR"

download_sequence() {
  local seq="$1"
  local dest="$CACHE_DIR/$seq"

  if [[ -d "$dest/mav0" ]]; then
    echo "[SKIP] $seq already cached at $dest"
    return 0
  fi

  echo "[INFO] Downloading $seq ..."
  mkdir -p "$dest"

  if [[ "$USE_ASL" == "1" ]]; then
    # Full dataset from ETH Research Collection (grouped ZIPs)
    local group
    group="$(seq_to_group "$seq")"
    if [[ -z "$group" ]]; then
      echo "[ERROR] Cannot determine group ZIP for sequence: $seq" >&2
      return 1
    fi
    local url="${ETH_GROUP_URLS[$group]}"
    # Cache the group ZIP so a second sequence in the same group skips the download.
    local group_zip="$CACHE_DIR/_group_${group}.zip"
    if [[ ! -f "$group_zip" ]]; then
      echo "[INFO] Downloading group ZIP for $group (~GB-scale) ..."
      wget --progress=bar:force -O "$group_zip" "$url"
    else
      echo "[INFO] Group ZIP already cached: $group_zip"
    fi
    # Step 1: extract the per-sequence ZIP from the group ZIP.
    # Layout inside group ZIP: <group>/<seq>/<seq>.zip  (plus a .bag)
    local seq_zip_inner="${group}/${seq}/${seq}.zip"
    local extract_dir="$CACHE_DIR/_extract_${seq}"
    echo "[INFO] Extracting $seq_zip_inner from $group_zip ..."
    unzip -q "$group_zip" "$seq_zip_inner" -d "$extract_dir"
    local seq_zip="$extract_dir/${seq_zip_inner}"
    if [[ ! -f "$seq_zip" ]]; then
      echo "[ERROR] $seq_zip_inner not found inside $group_zip" >&2
      rm -rf "$extract_dir"
      return 1
    fi

    # Step 2: extract the sequence ZIP itself — produces mav0/ at the root.
    echo "[INFO] Extracting $seq.zip ..."
    unzip -q "$seq_zip" -d "$dest"
    rm -rf "$extract_dir"
  else
    # Trimmed subset from GitHub Releases
    local url="$GH_RELEASE_BASE/${seq}_60s.tar.zst"
    local archive="$CACHE_DIR/${seq}_60s.tar.zst"
    wget --progress=bar:force -O "$archive" "$url" || {
      echo "[WARN] GitHub Release not found for $seq — falling back to ASL full dataset"
      USE_ASL=1 download_sequence "$seq"
      return
    }
    echo "[INFO] Extracting $archive ..."
    tar --zstd -xf "$archive" -C "$dest"
    rm -f "$archive"
  fi

  echo "[DONE] $seq -> $dest"
}

for seq in "${SEQUENCES[@]}"; do
  download_sequence "$seq"
done

echo ""
echo "All requested sequences ready in: $CACHE_DIR"
echo "Paths:"
for seq in "${SEQUENCES[@]}"; do
  echo "  $seq  ->  $CACHE_DIR/$seq"
done
