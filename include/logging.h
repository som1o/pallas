#ifndef LOGGING_H
#define LOGGING_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace spdlog {
namespace level {
enum level_enum {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    critical = 5
};
}  // namespace level
}  // namespace spdlog

namespace logging {

bool init_logging(const std::string& log_dir, const std::string& app_name);
void flush();

void log_event(spdlog::level::level_enum level,
               const std::string& event,
               const std::vector<std::pair<std::string, std::string>>& fields = {});

}  // namespace logging

#endif
