#include "system_metrics.h"

#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

bool read_cpu_totals(uint64_t& total, uint64_t& idle) {
    std::ifstream in("/proc/stat");
    if (!in) {
        return false;
    }

    std::string cpu;
    uint64_t user = 0;
    uint64_t nice = 0;
    uint64_t system = 0;
    uint64_t idle_ticks = 0;
    uint64_t iowait = 0;
    uint64_t irq = 0;
    uint64_t softirq = 0;
    uint64_t steal = 0;
    if (!(in >> cpu >> user >> nice >> system >> idle_ticks >> iowait >> irq >> softirq >> steal)) {
        return false;
    }

    idle = idle_ticks + iowait;
    total = user + nice + system + idle_ticks + iowait + irq + softirq + steal;
    return true;
}

double read_rss_mb() {
    std::ifstream in("/proc/self/status");
    if (!in) {
        return 0.0;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            double kb = 0.0;
            std::string unit;
            iss >> key >> kb >> unit;
            return kb / 1024.0;
        }
    }
    return 0.0;
}

}  // namespace

HardwareMonitor::HardwareMonitor() : prev_total_(0), prev_idle_(0) {
    read_cpu_totals(prev_total_, prev_idle_);
}

HardwareUtilization HardwareMonitor::sample() {
    HardwareUtilization out;

    uint64_t total = 0;
    uint64_t idle = 0;
    if (read_cpu_totals(total, idle) && total > prev_total_) {
        const uint64_t total_delta = total - prev_total_;
        const uint64_t idle_delta = idle - prev_idle_;
        out.cpu_percent = 100.0 * static_cast<double>(total_delta - idle_delta) / static_cast<double>(total_delta);
        prev_total_ = total;
        prev_idle_ = idle;
    }

    out.rss_mb = read_rss_mb();
    return out;
}
