#include "logging.h"

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

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "?";
                } else {
                    out += c;
                }
        }
    }
    return out;
}

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

const char* level_name(spdlog::level::level_enum level) {
    switch (level) {
        case spdlog::level::trace: return "trace";
        case spdlog::level::debug: return "debug";
        case spdlog::level::info: return "info";
        case spdlog::level::warn: return "warn";
        case spdlog::level::err: return "error";
        case spdlog::level::critical: return "critical";
        default: return "info";
    }
}

}  // namespace

namespace logging {

bool init_logging(const std::string& log_dir, const std::string& app_name) {
    std::lock_guard<std::mutex> lock(g_mu);
    try {
        std::filesystem::create_directories(log_dir);
        const std::string log_path = log_dir + "/" + app_name + ".log";
        g_file.open(log_path, std::ios::out | std::ios::app);
        g_ready = static_cast<bool>(g_file);
        return true;
    } catch (...) {
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

void log_event(spdlog::level::level_enum level,
               const std::string& event,
               const std::vector<std::pair<std::string, std::string>>& fields) {
    if (!g_ready) {
        return;
    }

    std::ostringstream oss;
    oss << "{"
        << "\"ts\":\"" << now_iso8601_utc() << "\"," 
        << "\"level\":\"" << level_name(level) << "\"," 
        << "\"event\":\"" << json_escape(event) << "\"";

    for (const auto& kv : fields) {
        oss << ",\"" << json_escape(kv.first) << "\":\"" << json_escape(kv.second) << "\"";
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
