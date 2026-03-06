#include "battle_server.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

namespace battle {

namespace {

constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxContentLength = 8 * 1024 * 1024;
constexpr int kSocketTimeoutSec = 5;

std::string status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        default: return "Internal Server Error";
    }
}

std::string trim_copy(const std::string& text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string to_lower_ascii(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string query_value(const std::string& path, const std::string& key) {
    const size_t q = path.find('?');
    if (q == std::string::npos) {
        return "";
    }
    std::string query = path.substr(q + 1);
    std::stringstream ss(query);
    std::string part;
    while (std::getline(ss, part, '&')) {
        const size_t eq = part.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        if (part.substr(0, eq) == key) {
            return part.substr(eq + 1);
        }
    }
    return "";
}

std::string url_decode(std::string_view encoded) {
    std::string out;
    out.reserve(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        const char c = encoded[i];
        if (c == '+') {
            out.push_back(' ');
            continue;
        }
        if (c == '%' && i + 2 < encoded.size()) {
            auto from_hex = [](char ch) -> int {
                if (ch >= '0' && ch <= '9') return ch - '0';
                if (ch >= 'a' && ch <= 'f') return 10 + (ch - 'a');
                if (ch >= 'A' && ch <= 'F') return 10 + (ch - 'A');
                return -1;
            };
            const int hi = from_hex(encoded[i + 1]);
            const int lo = from_hex(encoded[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(c);
    }
    return out;
}

std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string trimmed = trim_copy(token);
        if (!trimmed.empty()) {
            out.push_back(trimmed);
        }
    }
    return out;
}

std::string strip_query(const std::string& path) {
    const size_t q = path.find('?');
    if (q == std::string::npos) {
        return path;
    }
    return path.substr(0, q);
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

}  // namespace

BattleServer::BattleServer(BattleEngine* engine, std::string web_root, uint16_t port)
    : engine_(engine), web_root_(std::move(web_root)), port_(port) {}

bool BattleServer::run() {
    if (engine_ == nullptr) {
        return false;
    }

    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        return false;
    }

    int reuse = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(server_fd);
        return false;
    }

    if (listen(server_fd, 16) < 0) {
        close(server_fd);
        return false;
    }

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        timeval timeout {};
        timeout.tv_sec = kSocketTimeoutSec;
        timeout.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        char buffer[8192];
        std::string method;
        std::string path;
        std::string version;

        int status = 200;
        std::string content_type = "application/json";
        std::string body;

        std::string raw_request;
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            const ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
            if (n <= 0) {
                status = (errno == EAGAIN || errno == EWOULDBLOCK) ? 408 : 400;
                break;
            }
            raw_request.append(buffer, static_cast<size_t>(n));
            if (raw_request.size() > kMaxHeaderBytes + kMaxContentLength) {
                status = 413;
                break;
            }
            header_end = raw_request.find("\r\n\r\n");
            if (header_end == std::string::npos && raw_request.size() > kMaxHeaderBytes) {
                status = 400;
                break;
            }
        }

        size_t content_len = 0;
        if (status == 200 && header_end != std::string::npos) {
            std::string header_block = raw_request.substr(0, header_end);
            std::stringstream header_stream(header_block);
            std::string request_line;
            std::getline(header_stream, request_line);
            if (!request_line.empty() && request_line.back() == '\r') {
                request_line.pop_back();
            }
            std::stringstream req(request_line);
            req >> method >> path >> version;
            if (method.empty() || path.empty() || version.empty()) {
                status = 400;
            }

            std::string header_line;
            while (status == 200 && std::getline(header_stream, header_line)) {
                if (!header_line.empty() && header_line.back() == '\r') {
                    header_line.pop_back();
                }
                const size_t colon = header_line.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const std::string key = to_lower_ascii(trim_copy(header_line.substr(0, colon)));
                const std::string value = trim_copy(header_line.substr(colon + 1));
                if (key == "content-length") {
                    try {
                        content_len = static_cast<size_t>(std::stoull(value));
                    } catch (...) {
                        status = 400;
                        break;
                    }
                    if (content_len > kMaxContentLength) {
                        status = 413;
                        break;
                    }
                }
            }

            if (status == 200) {
                const size_t body_start = header_end + 4;
                body = raw_request.substr(body_start);
                if (body.size() > content_len) {
                    body.resize(content_len);
                }
                while (body.size() < content_len) {
                    const size_t remaining = content_len - body.size();
                    const size_t chunk = std::min(remaining, sizeof(buffer));
                    const ssize_t more = recv(client_fd, buffer, chunk, 0);
                    if (more <= 0) {
                        status = (errno == EAGAIN || errno == EWOULDBLOCK) ? 408 : 400;
                        break;
                    }
                    body.append(buffer, static_cast<size_t>(more));
                }
            }
        }

        std::string response_body;
        if (status == 200) {
            response_body = handle_request(method, path, body, &status, &content_type);
        } else {
            content_type = "application/json";
            if (status == 413) {
                response_body = "{\"error\":\"payload too large\"}";
            } else if (status == 408) {
                response_body = "{\"error\":\"request timeout\"}";
            } else {
                response_body = "{\"error\":\"bad request\"}";
            }
        }

        std::ostringstream resp;
        resp << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Cache-Control: no-cache\r\n";
        resp << "Content-Length: " << response_body.size() << "\r\n\r\n";
        resp << response_body;

        const std::string response = resp.str();
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);
    }

    close(server_fd);
    return true;
}

std::string BattleServer::content_type_for(const std::string& path) const {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".html") {
        return "text/html";
    }
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".css") {
        return "text/css";
    }
    if (path.size() >= 3 && path.substr(path.size() - 3) == ".js") {
        return "application/javascript";
    }
    return "text/plain";
}

std::string BattleServer::read_text_file(const std::string& path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string BattleServer::handle_request(const std::string& method,
                                         const std::string& path,
                                         const std::string& body,
                                         int* status,
                                         std::string* content_type) {
    const std::string clean_path = strip_query(path);

    if (clean_path == "/api/state") {
        if (method != "GET") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        *status = 200;
        *content_type = "application/json";
        return engine_->current_state_json();
    }

    if (clean_path == "/api/control/step") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        std::string readiness_error;
        if (!engine_->validate_model_readiness(&readiness_error)) {
            *status = 400;
            return "{\"error\":\"" + json_escape(readiness_error) + "\"}";
        }
        engine_->step_once();
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/start") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        std::string readiness_error;
        if (!engine_->validate_model_readiness(&readiness_error)) {
            *status = 400;
            return "{\"error\":\"" + json_escape(readiness_error) + "\"}";
        }
        engine_->start();
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/pause") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        engine_->pause();
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/end") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        engine_->end_battle();
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/reset") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        engine_->reset_battle();
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/speed") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        const std::string value = query_value(path, "ticks_per_second");
        if (!value.empty()) {
            try {
                engine_->set_tick_rate(std::stod(value));
            } catch (...) {
                *status = 400;
                return "{\"error\":\"invalid speed\"}";
            }
        }
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/duration") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }

        std::string error;
        const std::string min_value = query_value(path, "min_seconds");
        const std::string max_value = query_value(path, "max_seconds");
        if (!min_value.empty() || !max_value.empty()) {
            if (min_value.empty() || max_value.empty()) {
                *status = 400;
                return "{\"error\":\"min_seconds and max_seconds must be provided together\"}";
            }
            try {
                const uint64_t min_seconds = static_cast<uint64_t>(std::stoull(min_value));
                const uint64_t max_seconds = static_cast<uint64_t>(std::stoull(max_value));
                if (!engine_->set_battle_duration_bounds_seconds(min_seconds, max_seconds, &error)) {
                    *status = 400;
                    return "{\"error\":\"" + json_escape(error) + "\"}";
                }
            } catch (...) {
                *status = 400;
                return "{\"error\":\"invalid min/max duration\"}";
            }
        }

        const std::string value = query_value(path, "seconds");
        if (!value.empty()) {
            try {
                const uint64_t seconds = static_cast<uint64_t>(std::stoull(value));
                if (!engine_->set_battle_duration_seconds(seconds, &error)) {
                    *status = 400;
                    return "{\"error\":\"" + json_escape(error) + "\"}";
                }
            } catch (...) {
                *status = 400;
                return "{\"error\":\"invalid duration\"}";
            }
        }
        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/control/override") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }

        ManualOverrideCommand command;
        const std::string actor_value = query_value(path, "actor_country_id");
        const std::string target_value = query_value(path, "target_country_id");
        const std::string strategy_value = to_lower_ascii(url_decode(query_value(path, "strategy")));
        command.terms_type = url_decode(query_value(path, "terms_type"));
        command.terms_details = url_decode(query_value(path, "terms_details"));

        if (actor_value.empty() || strategy_value.empty()) {
            *status = 400;
            return "{\"error\":\"actor_country_id and strategy are required\"}";
        }

        try {
            command.actor_country_id = static_cast<uint16_t>(std::stoul(actor_value));
            if (!target_value.empty()) {
                command.target_country_id = static_cast<uint16_t>(std::stoul(target_value));
            }
        } catch (...) {
            *status = 400;
            return "{\"error\":\"invalid actor/target country id\"}";
        }

        if (strategy_value == "attack") {
            command.strategy = Strategy::Attack;
        } else if (strategy_value == "defend") {
            command.strategy = Strategy::Defend;
        } else if (strategy_value == "negotiate") {
            command.strategy = Strategy::Negotiate;
        } else if (strategy_value == "surrender") {
            command.strategy = Strategy::Surrender;
        } else if (strategy_value == "transfer_weapons") {
            command.strategy = Strategy::TransferWeapons;
        } else if (strategy_value == "focus_economy") {
            command.strategy = Strategy::FocusEconomy;
        } else if (strategy_value == "develop_technology") {
            command.strategy = Strategy::DevelopTechnology;
        } else if (strategy_value == "form_alliance") {
            command.strategy = Strategy::FormAlliance;
        } else if (strategy_value == "betray") {
            command.strategy = Strategy::Betray;
        } else if (strategy_value == "cyber_operation") {
            command.strategy = Strategy::CyberOperation;
        } else if (strategy_value == "sign_trade_agreement") {
            command.strategy = Strategy::SignTradeAgreement;
        } else if (strategy_value == "cancel_trade_agreement") {
            command.strategy = Strategy::CancelTradeAgreement;
        } else if (strategy_value == "impose_embargo") {
            command.strategy = Strategy::ImposeEmbargo;
        } else if (strategy_value == "invest_in_resource_extraction") {
            command.strategy = Strategy::InvestInResourceExtraction;
        } else if (strategy_value == "reduce_military_upkeep") {
            command.strategy = Strategy::ReduceMilitaryUpkeep;
        } else if (strategy_value == "suppress_dissent") {
            command.strategy = Strategy::SuppressDissent;
        } else if (strategy_value == "hold_elections") {
            command.strategy = Strategy::HoldElections;
        } else if (strategy_value == "coup_attempt") {
            command.strategy = Strategy::CoupAttempt;
        } else if (strategy_value == "propose_defense_pact") {
            command.strategy = Strategy::ProposeDefensePact;
        } else if (strategy_value == "propose_non_aggression") {
            command.strategy = Strategy::ProposeNonAggression;
        } else if (strategy_value == "break_treaty") {
            command.strategy = Strategy::BreakTreaty;
        } else if (strategy_value == "request_intel") {
            command.strategy = Strategy::RequestIntel;
        } else if (strategy_value == "deploy_units") {
            command.strategy = Strategy::DeployUnits;
        } else if (strategy_value == "tactical_nuke") {
            command.strategy = Strategy::TacticalNuke;
        } else if (strategy_value == "strategic_nuke") {
            command.strategy = Strategy::StrategicNuke;
        } else if (strategy_value == "cyber_attack") {
            command.strategy = Strategy::CyberAttack;
        } else {
            *status = 400;
            return "{\"error\":\"unknown strategy\"}";
        }

        std::string error;
        if (!engine_->apply_manual_override(command, &error)) {
            *status = 400;
            return "{\"error\":\"" + json_escape(error) + "\"}";
        }

        *status = 200;
        *content_type = "application/json";
        return "{\"ok\":true}";
    }

    if (clean_path == "/api/leaderboard") {
        if (method != "GET") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        *status = 200;
        *content_type = "application/json";
        return engine_->current_leaderboard_json();
    }

    if (clean_path == "/api/models") {
        if (method != "GET") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        *status = 200;
        *content_type = "application/json";
        return engine_->available_models_json();
    }

    if (clean_path == "/api/diagnostics") {
        if (method != "GET") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        *status = 200;
        *content_type = "application/json";
        return engine_->current_diagnostics_json();
    }

    if (clean_path == "/api/upload-model") {
        if (method != "POST") {
            *status = 405;
            return "{\"error\":\"method not allowed\"}";
        }
        const std::string model_name = url_decode(query_value(path, "name"));
        const std::string team_name = url_decode(query_value(path, "team"));
        const std::string country_value = query_value(path, "country_id");
        const std::string upload_label = url_decode(query_value(path, "label"));
        if (model_name.empty() && team_name.empty() && country_value.empty()) {
            *status = 400;
            return "{\"error\":\"missing model name, team, or country_id\"}";
        }
        if (body.empty()) {
            *status = 400;
            return "{\"error\":\"empty upload body\"}";
        }

        uint16_t country_id = 0;
        if (!country_value.empty()) {
            try {
                country_id = static_cast<uint16_t>(std::stoul(country_value));
            } catch (...) {
                *status = 400;
                return "{\"error\":\"invalid country_id\"}";
            }
        }

        std::string error;
        std::string applied_model;
        if (!engine_->upload_model_binary(model_name, team_name, country_id, upload_label, body, &error, &applied_model)) {
            std::cerr << "upload_model_failed"
                      << " model_name=" << model_name
                      << " team_name=" << team_name
                      << " country_id=" << country_id
                      << " label=" << upload_label
                      << " error=" << error
                      << '\n';
            *status = 400;
            *content_type = "application/json";
            return "{\"error\":\"" + json_escape(error) + "\"}";
        }
        std::cerr << "upload_model_applied"
                  << " model_name=" << model_name
                  << " team_name=" << team_name
                  << " country_id=" << country_id
                  << " label=" << upload_label
                  << " applied_model=" << applied_model
                  << '\n';
        *status = 200;
        *content_type = "application/json";
        std::ostringstream oss;
        oss << "{\"ok\":true,\"applied_model\":\"" << json_escape(applied_model) << "\",\"applied_models\":[";
        const std::vector<std::string> applied = split_csv(applied_model);
        for (size_t i = 0; i < applied.size(); ++i) {
            if (i > 0) {
                oss << ',';
            }
            oss << "\"" << json_escape(applied[i]) << "\"";
        }
        oss << "]}";
        return oss.str();
    }

    if (method != "GET") {
        *status = 405;
        *content_type = "application/json";
        return "{\"error\":\"method not allowed\"}";
    }

    std::string static_path = clean_path;
    if (static_path == "/") {
        static_path = "/index.html";
    }

    if (!static_path.empty() && static_path.front() == '/') {
        static_path.erase(static_path.begin());
    }

    const std::filesystem::path path_obj = std::filesystem::path(web_root_) / static_path;
    const std::string content = read_text_file(path_obj.string());
    if (content.empty()) {
        *status = 404;
        *content_type = "application/json";
        return "{\"error\":\"not found\"}";
    }

    *status = 200;
    *content_type = content_type_for(path_obj.string());
    return content;
}

}  // namespace battle
