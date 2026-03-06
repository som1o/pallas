#include "perf_counters.h"

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>

namespace {

long perf_event_open(struct perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

std::string read_perf_paranoid_value() {
    std::ifstream in("/proc/sys/kernel/perf_event_paranoid");
    std::string value;
    if (in) {
        std::getline(in, value);
    }
    return value;
}

}  // namespace

int PerfCounters::open_counter(uint32_t type, uint64_t config) {
    perf_event_attr pe;
    std::memset(&pe, 0, sizeof(perf_event_attr));
    pe.type = type;
    pe.size = sizeof(perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    const int fd = static_cast<int>(perf_event_open(&pe, 0, -1, -1, 0));
    if (fd < 0 && availability_reason_.empty()) {
        availability_reason_ = std::strerror(errno);
    }
    return fd;
}

uint64_t PerfCounters::now_monotonic_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count());
}

uint64_t PerfCounters::read_counter(int fd) {
    uint64_t value = 0;
    if (fd < 0) {
        return 0;
    }
    if (::read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) {
        return 0;
    }
    return value;
}

PerfCounters::PerfCounters()
    : fd_cycles_(open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)),
      fd_instr_(open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)),
      fd_cache_ref_(open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES)),
      fd_cache_miss_(open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES)),
      running_(false),
      available_(fd_cycles_ >= 0 && fd_instr_ >= 0 && fd_cache_ref_ >= 0 && fd_cache_miss_ >= 0),
      start_ns_(0) {}

PerfCounters::~PerfCounters() {
    if (fd_cycles_ >= 0) close(fd_cycles_);
    if (fd_instr_ >= 0) close(fd_instr_);
    if (fd_cache_ref_ >= 0) close(fd_cache_ref_);
    if (fd_cache_miss_ >= 0) close(fd_cache_miss_);
}

bool PerfCounters::available() const {
    return available_;
}

const std::string& PerfCounters::availability_reason() const {
    return availability_reason_;
}

bool PerfCounters::start() {
    if (!available_) {
        if (availability_reason_.empty()) {
            availability_reason_ = "perf counters unavailable";
        }
        const std::string paranoid = read_perf_paranoid_value();
        if (!paranoid.empty()) {
            availability_reason_ += "; perf_event_paranoid=" + paranoid;
        }
        return false;
    }

    ioctl(fd_cycles_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_instr_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_cache_ref_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_cache_miss_, PERF_EVENT_IOC_RESET, 0);

    ioctl(fd_cycles_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_instr_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_cache_ref_, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_cache_miss_, PERF_EVENT_IOC_ENABLE, 0);

    start_ns_ = now_monotonic_ns();
    running_ = true;
    return true;
}

PerfSnapshot PerfCounters::stop(size_t estimated_flops_per_sample, size_t sample_count) {
    PerfSnapshot out;
    out.available = available_;

    const uint64_t end_ns = now_monotonic_ns();
    if (running_) {
        ioctl(fd_cycles_, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_instr_, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_cache_ref_, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_cache_miss_, PERF_EVENT_IOC_DISABLE, 0);
        running_ = false;
    }

    out.elapsed_sec = static_cast<double>(end_ns - start_ns_) / 1e9;
    if (!available_) {
        return out;
    }

    out.cycles = read_counter(fd_cycles_);
    out.instructions = read_counter(fd_instr_);
    out.cache_references = read_counter(fd_cache_ref_);
    out.cache_misses = read_counter(fd_cache_miss_);

    const double total_flops = static_cast<double>(estimated_flops_per_sample) * static_cast<double>(sample_count);
    out.estimated_flops = out.elapsed_sec > 0.0 ? (total_flops / out.elapsed_sec) : 0.0;

    const double approx_bytes = static_cast<double>(out.cache_references) * 64.0;
    out.estimated_mem_bw_gbps = out.elapsed_sec > 0.0 ? (approx_bytes / out.elapsed_sec / 1e9) : 0.0;
    return out;
}
