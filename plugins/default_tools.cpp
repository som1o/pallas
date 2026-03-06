#include "tool_registry.h"
#include "logging.h"

namespace {

void attack_tool(const PluginToolContextV1* ctx) {
    const char* sector = (ctx == nullptr || ctx->sector == nullptr) ? "" : ctx->sector;
    const char* action = (ctx == nullptr || ctx->action == nullptr) ? "" : ctx->action;
    const int priority = (ctx == nullptr) ? 0 : ctx->priority;
    logging::log_event(logging::Level::Info, "plugin_attack", {
        {"sector", sector},
        {"action", action},
        {"priority", std::to_string(priority)}
    });
}

void defend_tool(const PluginToolContextV1* ctx) {
    const char* sector = (ctx == nullptr || ctx->sector == nullptr) ? "" : ctx->sector;
    const char* action = (ctx == nullptr || ctx->action == nullptr) ? "" : ctx->action;
    const int priority = (ctx == nullptr) ? 0 : ctx->priority;
    logging::log_event(logging::Level::Info, "plugin_defend", {
        {"sector", sector},
        {"action", action},
        {"priority", std::to_string(priority)}
    });
}

void retreat_tool(const PluginToolContextV1* ctx) {
    const char* sector = (ctx == nullptr || ctx->sector == nullptr) ? "" : ctx->sector;
    const char* action = (ctx == nullptr || ctx->action == nullptr) ? "" : ctx->action;
    const int priority = (ctx == nullptr) ? 0 : ctx->priority;
    logging::log_event(logging::Level::Info, "plugin_retreat", {
        {"sector", sector},
        {"action", action},
        {"priority", std::to_string(priority)}
    });
}

}  // namespace

extern "C" bool register_plugin_tools_v1(PluginRegisterToolFnV1 register_tool, void* user_data) {
    if (register_tool == nullptr) {
        return false;
    }
    bool ok = true;
    ok = register_tool("attack", &attack_tool, user_data) && ok;
    ok = register_tool("defend", &defend_tool, user_data) && ok;
    ok = register_tool("retreat", &retreat_tool, user_data) && ok;
    return ok;
}
