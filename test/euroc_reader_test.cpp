// Copyright (c) 2026, Long Vuong
// SPDX-License-Identifier: BSD-3-Clause

#include "ekf_vio/euroc_reader.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

// Minimal EuRoC-layout fixture under a temp directory.
class EurocFixture {
 public:
  EurocFixture() {
    root_ = fs::temp_directory_path() / ("ekf_vio_euroc_test_" + std::to_string(::getpid()));
    fs::create_directories(root_ / "mav0" / "cam0" / "data");
    fs::create_directories(root_ / "mav0" / "cam1" / "data");
    fs::create_directories(root_ / "mav0" / "imu0");
  }

  ~EurocFixture() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  const fs::path& root() const { return root_; }

  void writeImu() {
    std::ofstream f(root_ / "mav0" / "imu0" / "data.csv");
    f << "#timestamp,wx,wy,wz,ax,ay,az\n";
    // 1 ms samples
    for (int i = 0; i < 5; ++i) {
      const int64_t t_ns = 1'000'000'000LL + i * 1'000'000LL;
      f << t_ns << ",0,0,0,0,0,9.81\n";
    }
  }

  void writeCamCsv(const std::string& cam, const std::vector<int64_t>& t_ns) {
    std::ofstream f(root_ / "mav0" / cam / "data.csv");
    f << "#timestamp,filename\n";
    for (auto t : t_ns) {
      const std::string name = std::to_string(t) + ".png";
      // Tiny valid PNG not required — loadStereo emptiness is OK for these tests.
      f << t << "," << name << "\n";
    }
  }

  void writeGt(const std::vector<int64_t>& t_ns) {
    fs::create_directories(root_ / "mav0" / "state_groundtruth_estimate0");
    std::ofstream f(root_ / "mav0" / "state_groundtruth_estimate0" / "data.csv");
    f << "#timestamp,p,q,v,bw,ba\n";
    for (size_t i = 0; i < t_ns.size(); ++i) {
      f << t_ns[i] << "," << (1.0 + static_cast<double>(i)) << ",0,0,"
        << "1,0,0,0,"
        << "0,0,0,"
        << "0,0,0,"
        << "0,0,0\n";
    }
  }

 private:
  fs::path root_;
};

}  // namespace

// Index pairing would associate cam0[1] with cam1[0] when cam1 drops the first
// frame — timestamp sync must skip the unmatched cam0 row instead.
TEST(EurocReaderTest, PairsStereoByTimestampNotIndex) {
  EurocFixture fx;
  fx.writeImu();
  // cam0: t=1.0, 1.1, 1.2 s (ns)
  fx.writeCamCsv("cam0", {1'000'000'000LL, 1'100'000'000LL, 1'200'000'000LL});
  // cam1: missing 1.0s — only 1.1 and 1.2
  fx.writeCamCsv("cam1", {1'100'000'000LL, 1'200'000'000LL});

  ekf_vio::EurocReader reader(fx.root().string());
  ASSERT_TRUE(reader.load());
  ASSERT_EQ(reader.numStereo(), 2u);

  // First matched pair must be t=1.1s, not a cross-wired 1.0/1.1 pair.
  const auto s0 = reader.loadStereo(0);
  EXPECT_NEAR(s0.timestamp, 1.1, 1e-9);
  const auto s1 = reader.loadStereo(1);
  EXPECT_NEAR(s1.timestamp, 1.2, 1e-9);
}

TEST(EurocReaderTest, ClosestGroundTruthRespectsMaxDt) {
  EurocFixture fx;
  fx.writeImu();
  fx.writeCamCsv("cam0", {1'000'000'000LL});
  fx.writeCamCsv("cam1", {1'000'000'000LL});
  // GT starts 1.5 s later (classic EuRoC gap)
  fx.writeGt({2'500'000'000LL});

  ekf_vio::EurocReader reader(fx.root().string());
  ASSERT_TRUE(reader.load());

  ekf_vio::GroundTruth gt;
  // Ungated: returns nearest (legacy)
  ASSERT_TRUE(reader.closestGroundTruth(1.0, gt, -1.0));
  EXPECT_NEAR(gt.timestamp, 2.5, 1e-9);
  EXPECT_NEAR(gt.p.x(), 1.0, 1e-9);

  // Gated at 50 ms: must reject the 1.5 s gap
  EXPECT_FALSE(reader.closestGroundTruth(1.0, gt, 0.05));

  // Within gate
  ASSERT_TRUE(reader.closestGroundTruth(2.5, gt, 0.05));
  EXPECT_NEAR(gt.p.x(), 1.0, 1e-9);
}

TEST(EurocReaderTest, LoadFailsWhenNoStereoPairsMatch) {
  EurocFixture fx;
  fx.writeImu();
  fx.writeCamCsv("cam0", {1'000'000'000LL});
  fx.writeCamCsv("cam1", {9'000'000'000LL});  // 8 s away — no match

  ekf_vio::EurocReader reader(fx.root().string());
  EXPECT_FALSE(reader.load());
}
