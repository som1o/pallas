#ifndef SYSTEM_METRICS_H
#define SYSTEM_METRICS_H

#include <cstdint>

struct HardwareUtilization {
    double cpu_percent = 0.0;
    double rss_mb = 0.0;
};

class HardwareMonitor {
public:
    HardwareMonitor();
    HardwareUtilization sample();

private:
    uint64_t prev_total_;
    uint64_t prev_idle_;
};

#endif
