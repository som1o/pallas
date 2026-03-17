#ifndef BATTLE_SERVER_H
#define BATTLE_SERVER_H

#include "battle_runtime.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace battle {

class BattleServer {
public:
    BattleServer(BattleEngine& engine, std::string web_root, uint16_t port);

    bool run();

private:
    struct RequestResult {
        int status = 200;
        std::string content_type = "application/json";
        std::string body;
    };

    void handle_client(int client_fd) const;
    bool is_control_endpoint(const std::string& clean_path) const;
    bool is_origin_allowed(const std::unordered_map<std::string, std::string>& headers) const;
    bool is_authorized(const std::unordered_map<std::string, std::string>& headers,
                       const std::string& path) const;
    void append_cors_headers(const std::unordered_map<std::string, std::string>& headers,
                             std::vector<std::pair<std::string, std::string>>& response_headers) const;

    std::string content_type_for(const std::string& path) const;
    std::string read_text_file(const std::string& path) const;
    RequestResult handle_request(
        const std::string& method,
        const std::string& path,
        const std::unordered_map<std::string, std::string>& headers,
        const std::string& body,
        std::vector<std::pair<std::string, std::string>>& response_headers) const;

    BattleEngine& engine_;
    std::string web_root_;
    std::filesystem::path canonical_web_root_;
    std::string allowed_origin_;
    std::string control_token_;
    size_t max_upload_bytes_;
    uint16_t port_;
};

}  // namespace battle

#endif
