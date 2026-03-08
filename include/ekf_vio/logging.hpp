/**
 * ekf_vio/logging.hpp — Thin spdlog wrapper for the ekf-vio project.
 *
 * Usage:
 *   #include <ekf_vio/logging.hpp>
 *   auto log = ekf_vio::get_logger();   // returns shared "ekf_vio" logger
 *   log->info("hello {}", 42);
 *
 * Call ekf_vio::init_logging(spdlog::level::debug) once in main() to set the
 * global level.  If never called, the default spdlog level (info) applies.
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>

namespace ekf_vio {

/// Initialise the shared logger.  Safe to call more than once.
inline void init_logging(spdlog::level::level_enum level = spdlog::level::info) {
  auto logger = spdlog::get("ekf_vio");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ekf_vio");
  }
  logger->set_level(level);
  logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
}

/// Return the shared logger (creates it on first call if init_logging was not
/// called yet).
inline std::shared_ptr<spdlog::logger> get_logger() {
  auto logger = spdlog::get("ekf_vio");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ekf_vio");
    logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
  }
  return logger;
}

} // namespace ekf_vio
