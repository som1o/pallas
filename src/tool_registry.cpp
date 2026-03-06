#include "tool_registry.h"
#include "logging.h"

#include <dlfcn.h>

#include <chrono>
#include <filesystem>

ToolRegistry::~ToolRegistry() {
    for (auto& plugin : plugins_) {
        if (plugin.handle != nullptr) {
            dlclose(plugin.handle);
        }
    }
}

bool ToolRegistry::register_tool(const std::string& name, ToolHandler handler) {
    if (name.empty() || !handler) {
        return false;
    }
    tools_[name] = std::move(handler);
    return true;
}

bool ToolRegistry::execute(const std::string& name, const ToolRequestContext& ctx) const {
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return false;
    }
    it->second(ctx);
    return true;
}

bool ToolRegistry::has_tool(const std::string& name) const {
    return tools_.find(name) != tools_.end();
}

size_t ToolRegistry::plugin_count() const {
    return plugins_.size();
}

bool ToolRegistry::register_from_plugin(const char* name, PluginToolCallbackV1 cb, void* user_data) {
    if (name == nullptr || cb == nullptr || user_data == nullptr) {
        return false;
    }

    auto* registry = static_cast<ToolRegistry*>(user_data);
    return registry->register_tool(name, [cb](const ToolRequestContext& ctx) {
        const auto now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        PluginToolContextV1 abi_ctx;
        abi_ctx.abi_version = kPluginAbiVersionV1;
        abi_ctx.sector = ctx.sector.c_str();
        abi_ctx.action = ctx.action.c_str();
        abi_ctx.priority = ctx.priority;
        abi_ctx.timestamp_ms = now_ms;
        abi_ctx.trace_id = "";
        cb(&abi_ctx);
    });
}

size_t ToolRegistry::load_plugins_from_directory(const std::string& directory_path) {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::exists(directory_path, ec) || !fs::is_directory(directory_path, ec)) {
        return 0;
    }

    size_t loaded = 0;
    for (const auto& entry : fs::directory_iterator(directory_path, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".so") {
            continue;
        }

        const std::string path = entry.path().string();
        void* handle = dlopen(path.c_str(), RTLD_NOW);
        if (handle == nullptr) {
            const char* err = dlerror();
            logging::log_event(spdlog::level::err, "plugin_load_failed", {
                {"path", path},
                {"error", err == nullptr ? "unknown" : std::string(err)}
            });
            continue;
        }

        dlerror();
        auto fn = reinterpret_cast<RegisterPluginToolsFnV1>(dlsym(handle, "register_plugin_tools_v1"));
        const char* sym_err = dlerror();
        if (sym_err != nullptr || fn == nullptr) {
            logging::log_event(spdlog::level::err, "plugin_symbol_missing", {
                {"path", path}
            });
            dlclose(handle);
            continue;
        }

        if (!fn(&ToolRegistry::register_from_plugin, this)) {
            logging::log_event(spdlog::level::err, "plugin_registration_failed", {
                {"path", path}
            });
            dlclose(handle);
            continue;
        }

        plugins_.push_back({handle, path});
        ++loaded;
    }
    return loaded;
}
