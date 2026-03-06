#include "logging.h"

#include "common_utils.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {

std::mutex g_mu;
std::ofstream g_file;
bool g_ready = false;

std::string now_iso8601_utc() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto time = clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_utc{};
    gmtime_r(&time, &tm_utc);

    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

const char* level_name(logging::Level level) {
    switch (level) {
        case logging::Level::Trace: return "trace";
        case logging::Level::Debug: return "debug";
        case logging::Level::Info: return "info";
        case logging::Level::Warn: return "warn";
        case logging::Level::Error: return "error";
        case logging::Level::Critical: return "critical";
        default: return "info";
    }
}

}  // namespace

namespace logging {

bool init_logging(const std::string& log_dir, const std::string& app_name) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_ready = false;
    try {
        std::filesystem::create_directories(log_dir);
        const std::string log_path = log_dir + "/" + app_name + ".log";
        if (g_file.is_open()) {
            g_file.close();
        }
        g_file.open(log_path, std::ios::out | std::ios::app);
        g_ready = static_cast<bool>(g_file);
        return g_ready;
    } catch (...) {
        g_ready = false;
        return false;
    }
}

void flush() {
    std::lock_guard<std::mutex> lock(g_mu);
    std::cout << std::flush;
    if (g_file) {
        g_file << std::flush;
    }
}

void log_event(Level level,
               const std::string& event,
               const std::vector<std::pair<std::string, std::string>>& fields) {
    if (!g_ready) {
        return;
    }

    std::ostringstream oss;
    oss << "{"
        << "\"ts\":\"" << now_iso8601_utc() << "\"," 
        << "\"level\":\"" << level_name(level) << "\"," 
        << "\"event\":\"" << pallas::util::json_escape(event) << "\"";

    for (const auto& kv : fields) {
        oss << ",\"" << pallas::util::json_escape(kv.first) << "\":\""
            << pallas::util::json_escape(kv.second) << "\"";
    }
    oss << "}";
    const std::string line = oss.str();

    std::lock_guard<std::mutex> lock(g_mu);
    std::cout << line << std::endl;
    if (g_file) {
        g_file << line << '\n';
    }
}

}  // namespace logging
