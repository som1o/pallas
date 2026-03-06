#ifndef PERF_COUNTERS_H
#define PERF_COUNTERS_H

#include <cstddef>
#include <cstdint>
#include <string>

struct PerfSnapshot {
    bool available = false;
    double elapsed_sec = 0.0;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t cache_references = 0;
    uint64_t cache_misses = 0;
    double estimated_flops = 0.0;
    double estimated_mem_bw_gbps = 0.0;
};

class PerfCounters {
public:
    PerfCounters();
    ~PerfCounters();

    bool start();
    PerfSnapshot stop(size_t estimated_flops_per_sample, size_t sample_count);
    bool available() const;
    const std::string& availability_reason() const;

private:
    int fd_cycles_;
    int fd_instr_;
    int fd_cache_ref_;
    int fd_cache_miss_;
    bool running_;
    bool available_;
    uint64_t start_ns_;
    std::string availability_reason_;

    int open_counter(uint32_t type, uint64_t config);
    static uint64_t now_monotonic_ns();
    static uint64_t read_counter(int fd);
};

#endif
