#ifndef LOGGING_H
#define LOGGING_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace logging {

enum class Level {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Critical = 5,
};

bool init_logging(const std::string& log_dir, const std::string& app_name);
void flush();

void log_event(Level level,
               const std::string& event,
               const std::vector<std::pair<std::string, std::string>>& fields = {});

}  // namespace logging

#endif
