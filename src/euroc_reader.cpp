#include "ekf_vio/euroc_reader.hpp"
#include <ekf_vio/logging.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace ekf_vio {

namespace fs = std::filesystem;

// Nanosecond integer timestamp → seconds (double)
static double ns_to_sec(int64_t ns) {
    return static_cast<double>(ns) * 1e-9;
}

// Trim leading/trailing whitespace
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
EurocReader::EurocReader(const std::string& sequence_path) {
    // Accept either "<seq>" or "<seq>/mav0"
    fs::path p(sequence_path);
    if (fs::exists(p / "mav0"))
        base_path_ = (p / "mav0").string();
    else
        base_path_ = p.string();
}

// ---------------------------------------------------------------------------
bool EurocReader::load() {
    if (!loadImu()) {
        get_logger()->warn("Failed to load IMU data");
        return false;
    }

    if (!loadCameraTimestamps(base_path_ + "/cam0", cam0_entries_)) {
        get_logger()->warn("Failed to load cam0 data");
        return false;
    }

    if (!loadCameraTimestamps(base_path_ + "/cam1", cam1_entries_)) {
        get_logger()->warn("Failed to load cam1 data");
        return false;
    }

    // Build stereo timestamps from cam0 (cam0 is the reference / left camera)
    stereo_timestamps_.reserve(cam0_entries_.size());
    for (const auto& [t, _] : cam0_entries_)
        stereo_timestamps_.push_back(t);

    // Ground truth is optional
    loadGroundTruth();

    buildTimeline();

    get_logger()->info("Loaded: {} IMU, {} stereo, {} GT, {} total events",
                       imu_data_.size(), stereo_timestamps_.size(),
                       ground_truth_.size(), events_.size());
    return true;
}

// ---------------------------------------------------------------------------
bool EurocReader::loadImu() {
    const std::string path = base_path_ + "/imu0/data.csv";
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ','))
            tokens.push_back(trim(token));

        if (tokens.size() < 7) continue;

        ImuData d;
        d.timestamp = ns_to_sec(std::stoll(tokens[0]));
        d.gyro  = { std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]) };
        d.accel = { std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]) };
        imu_data_.push_back(d);
    }

    return !imu_data_.empty();
}

// ---------------------------------------------------------------------------
bool EurocReader::loadCameraTimestamps(
        const std::string& cam_dir,
        std::vector<std::pair<double, std::string>>& out) {

    const std::string csv_path = cam_dir + "/data.csv";
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string ts_str, filename;
        if (!std::getline(ss, ts_str, ',')) continue;
        if (!std::getline(ss, filename, ',')) continue;

        ts_str   = trim(ts_str);
        filename = trim(filename);

        double t = ns_to_sec(std::stoll(ts_str));
        std::string img_path = cam_dir + "/data/" + filename;
        out.emplace_back(t, img_path);
    }

    return !out.empty();
}

// ---------------------------------------------------------------------------
bool EurocReader::loadGroundTruth() {
    const std::string path = base_path_ + "/state_groundtruth_estimate0/data.csv";
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;  // optional — not an error

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ','))
            tokens.push_back(trim(token));

        // EuRoC GT format:
        //   timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z,
        //   v_x, v_y, v_z, bw_x, bw_y, bw_z, ba_x, ba_y, ba_z
        if (tokens.size() < 17) continue;

        GroundTruth gt;
        gt.timestamp = ns_to_sec(std::stoll(tokens[0]));
        gt.p   = { std::stod(tokens[1]), std::stod(tokens[2]),  std::stod(tokens[3]) };
        gt.q   = Eigen::Quaterniond(
                     std::stod(tokens[4]),   // w
                     std::stod(tokens[5]),   // x
                     std::stod(tokens[6]),   // y
                     std::stod(tokens[7]));  // z
        gt.v   = { std::stod(tokens[8]),  std::stod(tokens[9]),  std::stod(tokens[10]) };
        gt.b_g = { std::stod(tokens[11]), std::stod(tokens[12]), std::stod(tokens[13]) };
        gt.b_a = { std::stod(tokens[14]), std::stod(tokens[15]), std::stod(tokens[16]) };
        ground_truth_.push_back(gt);
    }

    return !ground_truth_.empty();
}

// ---------------------------------------------------------------------------
void EurocReader::buildTimeline() {
    events_.clear();
    events_.reserve(imu_data_.size() + stereo_timestamps_.size());

    for (size_t i = 0; i < imu_data_.size(); ++i)
        events_.push_back({DataEvent::IMU, i});

    for (size_t i = 0; i < stereo_timestamps_.size(); ++i)
        events_.push_back({DataEvent::STEREO, i});

    // Sort by timestamp
    std::sort(events_.begin(), events_.end(),
              [this](const DataEvent& a, const DataEvent& b) {
                  double ta = (a.type == DataEvent::IMU)
                              ? imu_data_[a.index].timestamp
                              : stereo_timestamps_[a.index];
                  double tb = (b.type == DataEvent::IMU)
                              ? imu_data_[b.index].timestamp
                              : stereo_timestamps_[b.index];
                  return ta < tb;
              });
}

// ---------------------------------------------------------------------------
StereoImages EurocReader::loadStereo(size_t stereo_index) const {
    StereoImages si;
    si.timestamp = stereo_timestamps_.at(stereo_index);

    si.left  = cv::imread(cam0_entries_[stereo_index].second,
                          cv::IMREAD_GRAYSCALE);
    si.right = cv::imread(cam1_entries_[stereo_index].second,
                          cv::IMREAD_GRAYSCALE);

    if (si.left.empty())
        get_logger()->warn("Failed to load left image: {}",
                           cam0_entries_[stereo_index].second);
    if (si.right.empty())
        get_logger()->warn("Failed to load right image: {}",
                           cam1_entries_[stereo_index].second);

    return si;
}

// ---------------------------------------------------------------------------
bool EurocReader::closestGroundTruth(double t, GroundTruth& out) const {
    if (ground_truth_.empty()) return false;

    // Binary search for closest timestamp
    auto it = std::lower_bound(
        ground_truth_.begin(), ground_truth_.end(), t,
        [](const GroundTruth& gt, double ts) { return gt.timestamp < ts; });

    if (it == ground_truth_.end()) {
        out = ground_truth_.back();
    } else if (it == ground_truth_.begin()) {
        out = ground_truth_.front();
    } else {
        auto prev = std::prev(it);
        out = (std::abs(it->timestamp - t) < std::abs(prev->timestamp - t))
              ? *it : *prev;
    }
    return true;
}

// ---------------------------------------------------------------------------
void EurocReader::replay(const ImuCallback& on_imu,
                         const StereoCallback& on_stereo) const {
    for (const auto& ev : events_) {
        if (ev.type == DataEvent::IMU) {
            on_imu(imu_data_[ev.index]);
        } else {
            on_stereo(loadStereo(ev.index));
        }
    }
}

} // namespace ekf_vio
