#ifndef TOOL_REGISTRY_H
#define TOOL_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

struct ToolRequestContext {
    std::string sector;
    std::string action;
    int priority = 0;
};

using ToolHandler = std::function<void(const ToolRequestContext&)>;

extern "C" {
constexpr uint32_t kPluginAbiVersionV1 = 1;

struct PluginToolContextV1 {
    uint32_t abi_version = kPluginAbiVersionV1;
    const char* sector = nullptr;
    const char* action = nullptr;
    int priority = 0;
    uint64_t timestamp_ms = 0;
    const char* trace_id = nullptr;
};

using PluginToolCallbackV1 = void(*)(const PluginToolContextV1*);
using PluginRegisterToolFnV1 = bool(*)(const char*, PluginToolCallbackV1, void*);
using RegisterPluginToolsFnV1 = bool(*)(PluginRegisterToolFnV1, void*);
}

class ToolRegistry {
public:
    ToolRegistry() = default;
    ~ToolRegistry();

    bool register_tool(const std::string& name, ToolHandler handler);
    bool execute(const std::string& name, const ToolRequestContext& ctx) const;
    bool has_tool(const std::string& name) const;

    size_t load_plugins_from_directory(const std::string& directory_path);
    size_t plugin_count() const;

private:
    struct PluginEntry {
        void* handle = nullptr;
        std::string path;
    };

    static bool register_from_plugin(const char* name, PluginToolCallbackV1 cb, void* user_data);

    std::unordered_map<std::string, ToolHandler> tools_;
    std::vector<PluginEntry> plugins_;
};

#endif
