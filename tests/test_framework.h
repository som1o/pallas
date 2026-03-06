#ifndef PALLAS_TEST_FRAMEWORK_H
#define PALLAS_TEST_FRAMEWORK_H

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace pallas_test {

struct Context {
    int failures = 0;
    int assertions = 0;
};

using TestFn = void (*)(Context& ctx);

struct TestCase {
    std::string name;
    TestFn fn = nullptr;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline bool register_test(const char* name, TestFn fn) {
    registry().push_back(TestCase{name, fn});
    return true;
}

inline void expect_true(Context& ctx, bool cond, const char* expr, const char* file, int line, const std::string& msg) {
    ++ctx.assertions;
    if (!cond) {
        ++ctx.failures;
        std::cerr << "[FAIL] " << file << ':' << line << " " << expr;
        if (!msg.empty()) {
            std::cerr << " :: " << msg;
        }
        std::cerr << '\n';
    }
}

inline int run_all() {
    Context ctx;
    const char* filter_env = std::getenv("PALLAS_TEST_FILTER");
    const std::string filter = filter_env == nullptr ? "" : std::string(filter_env);

    int executed = 0;
    for (const auto& t : registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) {
            continue;
        }
        ++executed;
        try {
            t.fn(ctx);
        } catch (const std::exception& ex) {
            ++ctx.failures;
            std::cerr << "[FAIL] " << t.name << " threw: " << ex.what() << '\n';
        } catch (...) {
            ++ctx.failures;
            std::cerr << "[FAIL] " << t.name << " threw unknown exception\n";
        }
    }

    if (executed == 0) {
        std::cerr << "No tests executed";
        if (!filter.empty()) {
            std::cerr << " (filter: " << filter << ')';
        }
        std::cerr << '\n';
        return 1;
    }

    if (ctx.failures > 0) {
        std::cerr << "Tests failed: " << ctx.failures << " / assertions=" << ctx.assertions << '\n';
        return 1;
    }

    std::cout << "All tests passed: " << executed << " test cases, assertions=" << ctx.assertions << '\n';
    return 0;
}

}  // namespace pallas_test

#define TEST_CASE(name)                                                     \
    static void name(pallas_test::Context&);                                \
    static const bool name##_registered = pallas_test::register_test(#name, &name); \
    static void name(pallas_test::Context& ctx)

#define EXPECT_TRUE(cond, msg) pallas_test::expect_true(ctx, (cond), #cond, __FILE__, __LINE__, (msg))

#endif
