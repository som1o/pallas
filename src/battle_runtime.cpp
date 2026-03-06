#include "battle_runtime.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_set>

namespace battle {

namespace {

constexpr uint32_t kReplayMagic = 0x50424C47;
constexpr uint32_t kReplayVersion = 6;
constexpr uint32_t kReplayChunkMagic = 0x43484E4B;  // CHNK

bool contains_id(const std::vector<uint16_t>& values, uint16_t id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

void add_unique_id(std::vector<uint16_t>* values, uint16_t id) {
    if (values == nullptr || id == 0 || contains_id(*values, id)) {
        return;
    }
    values->push_back(id);
}

void erase_id(std::vector<uint16_t>* values, uint16_t id) {
    if (values == nullptr) {
        return;
    }
    values->erase(std::remove(values->begin(), values->end(), id), values->end());
}

std::vector<char> rle_compress(const std::vector<char>& input) {
    std::vector<char> out;
    out.reserve(input.size());
    size_t i = 0;
    while (i < input.size()) {
        const char value = input[i];
        uint8_t run = 1;
        while (i + run < input.size() && input[i + run] == value && run < 255) {
            ++run;
        }
        out.push_back(static_cast<char>(run));
        out.push_back(value);
        i += run;
    }
    return out;
}

bool rle_decompress(const std::vector<char>& input, std::vector<char>* output, size_t expected_size) {
    if (output == nullptr) {
        return false;
    }
    output->clear();
    output->reserve(expected_size);
    if (input.size() % 2 != 0) {
        return false;
    }
    for (size_t i = 0; i < input.size(); i += 2) {
        const uint8_t run = static_cast<uint8_t>(input[i]);
        const char value = input[i + 1];
        output->insert(output->end(), run, value);
    }
    return output->size() == expected_size;
}

std::string decision_signature(const DecisionEnvelope& d) {
    std::ostringstream oss;
    oss << d.model_name << '|'
        << d.decision.actor_country_id << '|'
        << d.decision.target_country_id << '|'
        << static_cast<uint32_t>(d.decision.strategy) << '|'
        << d.decision.force_commitment << '|'
        << d.decision.terms.type << '|'
        << d.decision.terms.details << '|'
        << (d.decision.has_secondary_action ? 1 : 0) << '|'
        << static_cast<uint32_t>(d.decision.secondary_action.strategy) << '|'
        << d.decision.secondary_action.target_country_id << '|'
        << d.decision.secondary_action.commitment << '|'
        << d.decision.secondary_action.terms.type << '|'
        << d.decision.secondary_action.terms.details;
    return oss.str();
}

bool validate_battle_model_dims(size_t input_dim, size_t output_dim, std::string* error_message) {
    if (input_dim == battle_common::kBattleInputDim && output_dim == battle_common::kBattleOutputDim) {
        return true;
    }
    if (error_message != nullptr) {
        *error_message = "battle models must use architecture " +
                         std::to_string(battle_common::kBattleInputDim) + "x" +
                         std::to_string(battle_common::kBattleOutputDim) +
                         ", got " + std::to_string(input_dim) + "x" +
                         std::to_string(output_dim);
    }
    return false;
}

template <typename T>
void write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool read_binary(std::ifstream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

int strategy_priority(Strategy strategy) {
    switch (strategy) {
        case Strategy::Surrender: return 0;
        case Strategy::Defend: return 1;
        case Strategy::Negotiate: return 2;
        case Strategy::SignTradeAgreement: return 2;
        case Strategy::CancelTradeAgreement: return 2;
        case Strategy::TransferWeapons: return 3;
        case Strategy::FocusEconomy: return 3;
        case Strategy::ImposeEmbargo: return 3;
        case Strategy::DevelopTechnology: return 4;
        case Strategy::FormAlliance: return 4;
        case Strategy::InvestInResourceExtraction: return 4;
        case Strategy::ReduceMilitaryUpkeep: return 4;
        case Strategy::SuppressDissent: return 5;
        case Strategy::HoldElections: return 5;
        case Strategy::Betray: return 6;
        case Strategy::CyberOperation: return 7;
        case Strategy::CoupAttempt: return 8;
        case Strategy::ProposeDefensePact: return 2;
        case Strategy::ProposeNonAggression: return 2;
        case Strategy::BreakTreaty: return 6;
        case Strategy::RequestIntel: return 4;
        case Strategy::DeployUnits: return 7;
        case Strategy::TacticalNuke: return 9;
        case Strategy::StrategicNuke: return 9;
        case Strategy::CyberAttack: return 8;
        case Strategy::Attack: return 9;
    }
    return 1;
}

const char* strategy_to_string(Strategy strategy) {
    switch (strategy) {
        case Strategy::Attack: return "attack";
        case Strategy::Defend: return "defend";
        case Strategy::Negotiate: return "negotiate";
        case Strategy::Surrender: return "surrender";
        case Strategy::TransferWeapons: return "transfer_weapons";
        case Strategy::FocusEconomy: return "focus_economy";
        case Strategy::DevelopTechnology: return "develop_technology";
        case Strategy::FormAlliance: return "form_alliance";
        case Strategy::Betray: return "betray";
        case Strategy::CyberOperation: return "cyber_operation";
        case Strategy::SignTradeAgreement: return "sign_trade_agreement";
        case Strategy::CancelTradeAgreement: return "cancel_trade_agreement";
        case Strategy::ImposeEmbargo: return "impose_embargo";
        case Strategy::InvestInResourceExtraction: return "invest_in_resource_extraction";
        case Strategy::ReduceMilitaryUpkeep: return "reduce_military_upkeep";
        case Strategy::SuppressDissent: return "suppress_dissent";
        case Strategy::HoldElections: return "hold_elections";
        case Strategy::CoupAttempt: return "coup_attempt";
        case Strategy::ProposeDefensePact: return "propose_defense_pact";
        case Strategy::ProposeNonAggression: return "propose_non_aggression";
        case Strategy::BreakTreaty: return "break_treaty";
        case Strategy::RequestIntel: return "request_intel";
        case Strategy::DeployUnits: return "deploy_units";
        case Strategy::TacticalNuke: return "tactical_nuke";
        case Strategy::StrategicNuke: return "strategic_nuke";
        case Strategy::CyberAttack: return "cyber_attack";
    }
    return "defend";
}

const char* stance_to_string(sim::DiplomaticStance stance) {
    switch (stance) {
        case sim::DiplomaticStance::Aggressive: return "aggressive";
        case sim::DiplomaticStance::Neutral: return "neutral";
        case sim::DiplomaticStance::Pacifist: return "pacifist";
    }
    return "neutral";
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
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

std::string sanitize_wire(const std::string& value) {
    std::string out = value;
    for (char& ch : out) {
        if (ch == '\n' || ch == '\r' || ch == '\t' || ch == '|') {
            ch = ' ';
        }
    }
    return out;
}

bool parse_endpoint(const std::string& endpoint, sockaddr_in* addr) {
    if (addr == nullptr) {
        return false;
    }
    const size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) {
        return false;
    }

    const std::string host = endpoint.substr(0, colon);
    const std::string port_text = endpoint.substr(colon + 1);
    uint16_t port = 0;
    try {
        port = static_cast<uint16_t>(std::stoul(port_text));
    } catch (...) {
        return false;
    }

    std::memset(addr, 0, sizeof(sockaddr_in));
    addr->sin_family = AF_INET;
    addr->sin_port = htons(port);

    if (::inet_pton(AF_INET, host.c_str(), &addr->sin_addr) == 1) {
        return true;
    }

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* result = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &result) != 0 || result == nullptr) {
        return false;
    }
    auto* ipv4 = reinterpret_cast<sockaddr_in*>(result->ai_addr);
    addr->sin_addr = ipv4->sin_addr;
    ::freeaddrinfo(result);
    return true;
}

std::string serialize_decision_packet(uint32_t node_id,
                                      uint64_t tick,
                                      const std::vector<DecisionEnvelope>& decisions) {
    std::ostringstream oss;
    oss << "PALLAS_DSYNC|1|" << node_id << "|" << tick << "|" << decisions.size() << "\n";
    for (const DecisionEnvelope& d : decisions) {
        oss << sanitize_wire(d.model_name) << '|'
            << sanitize_wire(d.team) << '|'
            << d.decision.actor_country_id << '|'
            << d.decision.target_country_id << '|'
            << static_cast<uint32_t>(d.decision.strategy) << '|'
            << d.decision.force_commitment << '|'
            << sanitize_wire(d.decision.terms.type) << '|'
            << sanitize_wire(d.decision.terms.details) << '|'
            << (d.decision.has_secondary_action ? 1 : 0) << '|'
            << static_cast<uint32_t>(d.decision.secondary_action.strategy) << '|'
            << d.decision.secondary_action.target_country_id << '|'
            << d.decision.secondary_action.commitment << '|'
            << sanitize_wire(d.decision.secondary_action.terms.type) << '|'
            << sanitize_wire(d.decision.secondary_action.terms.details) << '\n';
    }
    return oss.str();
}

bool deserialize_decision_packet(const std::string& packet,
                                 uint32_t* out_node_id,
                                 uint64_t* out_tick,
                                 std::vector<DecisionEnvelope>* out_decisions) {
    if (out_node_id == nullptr || out_tick == nullptr || out_decisions == nullptr) {
        return false;
    }

    std::stringstream ss(packet);
    std::string header;
    if (!std::getline(ss, header)) {
        return false;
    }

    std::stringstream hs(header);
    std::vector<std::string> hp;
    std::string part;
    while (std::getline(hs, part, '|')) {
        hp.push_back(part);
    }
    if (hp.size() < 5 || hp[0] != "PALLAS_DSYNC") {
        return false;
    }

    try {
        *out_node_id = static_cast<uint32_t>(std::stoul(hp[2]));
        *out_tick = static_cast<uint64_t>(std::stoull(hp[3]));
    } catch (...) {
        return false;
    }

    out_decisions->clear();
    while (std::getline(ss, part)) {
        if (part.empty()) {
            continue;
        }
        std::stringstream ds(part);
        std::vector<std::string> cols;
        std::string c;
        while (std::getline(ds, c, '|')) {
            cols.push_back(c);
        }
        if (cols.size() < 14) {
            continue;
        }

        DecisionEnvelope d;
        d.model_name = cols[0];
        d.team = cols[1];
        try {
            d.decision.actor_country_id = static_cast<uint16_t>(std::stoul(cols[2]));
            d.decision.target_country_id = static_cast<uint16_t>(std::stoul(cols[3]));
            d.decision.strategy = static_cast<Strategy>(std::stoul(cols[4]));
            d.decision.force_commitment = std::stof(cols[5]);
        } catch (...) {
            continue;
        }
        d.decision.terms.type = cols[6];
        d.decision.terms.details = cols[7];
        d.decision.has_secondary_action = cols[8] == "1";
        try {
            d.decision.secondary_action.strategy = static_cast<Strategy>(std::stoul(cols[9]));
            d.decision.secondary_action.target_country_id = static_cast<uint16_t>(std::stoul(cols[10]));
            d.decision.secondary_action.commitment = std::stof(cols[11]);
        } catch (...) {
            d.decision.has_secondary_action = false;
        }
        d.decision.secondary_action.terms.type = cols[12];
        d.decision.secondary_action.terms.details = cols[13];
        out_decisions->push_back(std::move(d));
    }
    return true;
}

std::shared_ptr<Model> load_configured_model(const std::string& state_path, std::string* error_message) {
    if (state_path.empty()) {
        return nullptr;
    }

    size_t input_dim = 0;
    size_t output_dim = 0;
    ModelConfig config;
    std::string inspect_error;
    if (!inspect_model_state(state_path, &input_dim, &output_dim, &config, &inspect_error)) {
        if (error_message != nullptr) {
            *error_message = "inspect failed for '" + state_path + "': " + inspect_error;
        }
        return nullptr;
    }

    if (input_dim == 0 || output_dim == 0) {
        if (error_message != nullptr) {
            *error_message = "invalid model architecture for '" + state_path + "'";
        }
        return nullptr;
    }

    if (!validate_battle_model_dims(input_dim, output_dim, error_message)) {
        return nullptr;
    }

    auto model = std::make_shared<Model>(input_dim, output_dim, config);
    if (!model->load_state(state_path, input_dim, output_dim, nullptr)) {
        if (error_message != nullptr) {
            *error_message = "failed to load model state: " + state_path;
        }
        return nullptr;
    }

    model->set_training(false);
    model->set_inference_only(true);
    return model;
}

}  // namespace

class DistributedDecisionBus {
public:
    struct Stats {
        uint64_t exchange_count = 0;
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint64_t packets_dropped = 0;
        uint32_t peer_count = 0;
        uint32_t total_nodes = 1;
    };

    DistributedDecisionBus() = default;
    ~DistributedDecisionBus() {
        close_socket();
    }

    bool configure(const DistributedRuntimeConfig& config, std::string* error_message) {
        std::lock_guard<std::mutex> lock(mu_);
        config_ = config;
        close_socket();
        peers_.clear();

        if (config_.total_nodes <= 1) {
            return true;
        }

        socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd_ < 0) {
            if (error_message != nullptr) {
                *error_message = "failed to create distributed UDP socket";
            }
            return false;
        }

        int reuse = 1;
        ::setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(config_.bind_port);
        if (::inet_pton(AF_INET, config_.bind_host.c_str(), &bind_addr.sin_addr) != 1) {
            bind_addr.sin_addr.s_addr = INADDR_ANY;
        }

        if (::bind(socket_fd_, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0) {
            if (error_message != nullptr) {
                *error_message = "failed to bind distributed UDP socket";
            }
            close_socket();
            return false;
        }

        const int flags = ::fcntl(socket_fd_, F_GETFL, 0);
        ::fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

        for (const std::string& endpoint : config_.peer_endpoints) {
            sockaddr_in addr{};
            if (parse_endpoint(endpoint, &addr)) {
                peers_.push_back(addr);
            }
        }
        return true;
    }

    void exchange(uint64_t tick,
                  const std::vector<DecisionEnvelope>& local_decisions,
                  std::vector<DecisionEnvelope>* out_remote_decisions) {
        if (out_remote_decisions == nullptr) {
            return;
        }
        out_remote_decisions->clear();

        std::lock_guard<std::mutex> lock(mu_);

        if (config_.total_nodes <= 1 || socket_fd_ < 0) {
            return;
        }

        exchange_counter_.fetch_add(1, std::memory_order_relaxed);
        const std::string payload = serialize_decision_packet(config_.node_id, tick, local_decisions);
        for (const sockaddr_in& peer : peers_) {
            ::sendto(socket_fd_, payload.data(), payload.size(), 0,
                     reinterpret_cast<const sockaddr*>(&peer), sizeof(peer));
            packets_sent_.fetch_add(1, std::memory_order_relaxed);
        }

        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(std::max<uint32_t>(5U, config_.receive_timeout_ms));
        std::unordered_set<uint32_t> seen_nodes;
        while (std::chrono::steady_clock::now() < deadline) {
            char buffer[65535];
            sockaddr_in src{};
            socklen_t src_len = sizeof(src);
            const ssize_t n = ::recvfrom(socket_fd_, buffer, sizeof(buffer), 0,
                                         reinterpret_cast<sockaddr*>(&src), &src_len);
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            uint32_t node_id = 0;
            uint64_t packet_tick = 0;
            std::vector<DecisionEnvelope> remote;
            if (!deserialize_decision_packet(std::string(buffer, static_cast<size_t>(n)), &node_id, &packet_tick, &remote)) {
                packets_dropped_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (node_id == config_.node_id || packet_tick != tick) {
                packets_dropped_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (seen_nodes.find(node_id) != seen_nodes.end()) {
                packets_dropped_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            seen_nodes.insert(node_id);
            packets_received_.fetch_add(1, std::memory_order_relaxed);
            out_remote_decisions->insert(out_remote_decisions->end(), remote.begin(), remote.end());
            if (seen_nodes.size() + 1 >= config_.total_nodes) {
                break;
            }
        }
    }

    Stats snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        Stats stats;
        stats.exchange_count = exchange_counter_.load(std::memory_order_relaxed);
        stats.packets_sent = packets_sent_.load(std::memory_order_relaxed);
        stats.packets_received = packets_received_.load(std::memory_order_relaxed);
        stats.packets_dropped = packets_dropped_.load(std::memory_order_relaxed);
        stats.peer_count = static_cast<uint32_t>(peers_.size());
        stats.total_nodes = std::max<uint32_t>(1U, config_.total_nodes);
        return stats;
    }

private:
    void close_socket() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
    }

    int socket_fd_ = -1;
    DistributedRuntimeConfig config_;
    std::vector<sockaddr_in> peers_;
    mutable std::mutex mu_;
    std::atomic<uint64_t> exchange_counter_{0};
    std::atomic<uint64_t> packets_sent_{0};
    std::atomic<uint64_t> packets_received_{0};
    std::atomic<uint64_t> packets_dropped_{0};
};

void ModelManager::add_model(const ManagedModel& managed_model) {
    models_.push_back(managed_model);
}

std::vector<DecisionEnvelope> ModelManager::gather_decisions(const sim::World& world) const {
    std::vector<DecisionEnvelope> out;
    const WorldSnapshot snapshot = build_world_snapshot(world);

    for (const ManagedModel& managed : models_) {
        if (!managed.model) {
            continue;
        }
        for (uint16_t country_id : managed.controlled_country_ids) {
            if (distributed_total_nodes_ > 1) {
                if (static_cast<uint32_t>(country_id % distributed_total_nodes_) != distributed_node_id_) {
                    continue;
                }
            }
            DecisionEnvelope envelope;
            envelope.model_name = managed.name;
            envelope.team = managed.team;
            envelope.decision = managed.model->decide(snapshot, country_id);
            out.push_back(std::move(envelope));
        }
    }

    std::sort(out.begin(), out.end(), [](const DecisionEnvelope& a, const DecisionEnvelope& b) {
        const int pa = strategy_priority(a.decision.strategy);
        const int pb = strategy_priority(b.decision.strategy);
        if (pa == pb) {
            return a.decision.actor_country_id < b.decision.actor_country_id;
        }
        return pa < pb;
    });
    return out;
}

void ModelManager::apply_decisions(sim::World& world, const std::vector<DecisionEnvelope>& decisions) const {
    std::unordered_set<uint32_t> negotiation_edges;
    std::unordered_set<uint32_t> trade_edges;
    std::unordered_set<uint32_t> defense_edges;
    std::unordered_set<uint32_t> non_aggression_edges;
    auto for_each_action = [&](const DecisionEnvelope& envelope, const auto& fn) {
        fn(envelope.decision);
        if (!envelope.decision.has_secondary_action) {
            return;
        }
        ModelDecision secondary = envelope.decision;
        secondary.strategy = envelope.decision.secondary_action.strategy;
        secondary.target_country_id = envelope.decision.secondary_action.target_country_id;
        secondary.terms = envelope.decision.secondary_action.terms;
        secondary.force_commitment = envelope.decision.secondary_action.commitment;
        secondary.has_secondary_action = false;
        fn(secondary);
    };
    negotiation_edges.reserve(decisions.size());
    trade_edges.reserve(decisions.size());
    defense_edges.reserve(decisions.size());
    non_aggression_edges.reserve(decisions.size());
    for (const DecisionEnvelope& envelope : decisions) {
        for_each_action(envelope, [&](const ModelDecision& action) {
            const uint16_t from = action.actor_country_id;
            const uint16_t to = action.target_country_id;
            if (from == 0 || to == 0 || from == to) {
                return;
            }
            if (action.strategy == Strategy::Negotiate) {
                negotiation_edges.insert((static_cast<uint32_t>(from) << 16U) | static_cast<uint32_t>(to));
            } else if (action.strategy == Strategy::SignTradeAgreement) {
                trade_edges.insert((static_cast<uint32_t>(from) << 16U) | static_cast<uint32_t>(to));
            } else if (action.strategy == Strategy::ProposeDefensePact) {
                defense_edges.insert((static_cast<uint32_t>(from) << 16U) | static_cast<uint32_t>(to));
            } else if (action.strategy == Strategy::ProposeNonAggression) {
                non_aggression_edges.insert((static_cast<uint32_t>(from) << 16U) | static_cast<uint32_t>(to));
            }
        });
    }

    auto has_reciprocal_negotiation = [&](uint16_t a, uint16_t b) -> bool {
        if (a == 0 || b == 0 || a == b) {
            return false;
        }
        const uint32_t forward = (static_cast<uint32_t>(a) << 16U) | static_cast<uint32_t>(b);
        const uint32_t reverse = (static_cast<uint32_t>(b) << 16U) | static_cast<uint32_t>(a);
        return negotiation_edges.find(forward) != negotiation_edges.end() &&
               negotiation_edges.find(reverse) != negotiation_edges.end();
    };

    auto has_reciprocal_trade = [&](uint16_t a, uint16_t b) -> bool {
        if (a == 0 || b == 0 || a == b) {
            return false;
        }
        const uint32_t forward = (static_cast<uint32_t>(a) << 16U) | static_cast<uint32_t>(b);
        const uint32_t reverse = (static_cast<uint32_t>(b) << 16U) | static_cast<uint32_t>(a);
        return trade_edges.find(forward) != trade_edges.end() && trade_edges.find(reverse) != trade_edges.end();
    };

    auto has_reciprocal_defense = [&](uint16_t a, uint16_t b) -> bool {
        if (a == 0 || b == 0 || a == b) {
            return false;
        }
        const uint32_t forward = (static_cast<uint32_t>(a) << 16U) | static_cast<uint32_t>(b);
        const uint32_t reverse = (static_cast<uint32_t>(b) << 16U) | static_cast<uint32_t>(a);
        return defense_edges.find(forward) != defense_edges.end() && defense_edges.find(reverse) != defense_edges.end();
    };

    auto has_reciprocal_non_aggression = [&](uint16_t a, uint16_t b) -> bool {
        if (a == 0 || b == 0 || a == b) {
            return false;
        }
        const uint32_t forward = (static_cast<uint32_t>(a) << 16U) | static_cast<uint32_t>(b);
        const uint32_t reverse = (static_cast<uint32_t>(b) << 16U) | static_cast<uint32_t>(a);
        return non_aggression_edges.find(forward) != non_aggression_edges.end() && non_aggression_edges.find(reverse) != non_aggression_edges.end();
    };

    for (const DecisionEnvelope& envelope : decisions) {
        for_each_action(envelope, [&](const ModelDecision& decision) {
        if (decision.actor_country_id == 0) {
            return;
        }

        sim::Country* actor = find_country(world, decision.actor_country_id);
        if (actor == nullptr) {
            return;
        }

        switch (decision.strategy) {
            case Strategy::Attack: {
                const uint16_t target = decision.target_country_id;
                if (target == 0 || find_country(world, target) == nullptr) {
                    return;
                }

                sim::Country* target_country = find_country(world, target);
                bool allied = false;
                for (uint16_t ally_id : actor->allied_country_ids) {
                    if (ally_id == target) {
                        allied = true;
                        break;
                    }
                }

                if (allied) {
                    actor->betrayal_tick_log.push_back(world.current_tick());
                    actor->allied_country_ids.erase(
                        std::remove(actor->allied_country_ids.begin(), actor->allied_country_ids.end(), target),
                        actor->allied_country_ids.end());
                    if (target_country != nullptr) {
                        target_country->allied_country_ids.erase(
                            std::remove(target_country->allied_country_ids.begin(), target_country->allied_country_ids.end(), decision.actor_country_id),
                            target_country->allied_country_ids.end());
                        target_country->civilian_morale = std::max(sim::Fixed::from_int(0), target_country->civilian_morale - sim::Fixed::from_double(2.0));
                    }
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(1.0));
                }

                const double terrain_hint = 1.0 - actor->terrain.mountains.to_double() * 0.10 + actor->terrain.forests.to_double() * 0.04;
                const double surprise_hint = 0.88 + actor->intelligence_level.to_double() / 360.0 + decision.force_commitment * 0.08;
                const uint32_t route_distance = static_cast<uint32_t>(
                    std::max<int>(1, static_cast<int>(actor->adjacent_country_ids.size()) + static_cast<int>(world.current_tick() % 3)));

                world.schedule_event(std::make_unique<sim::AttackEvent>(
                                       decision.actor_country_id,
                                       target,
                                       sim::Fixed::from_double(std::clamp(terrain_hint, 0.70, 1.18)),
                                       sim::Fixed::from_double(std::clamp(surprise_hint, 0.70, 1.25)),
                                       route_distance),
                                   world.current_tick());

                const int64_t offensive_window = std::clamp<int64_t>(actor->logistics_capacity.to_int() / 35 + static_cast<int64_t>(decision.force_commitment * 2.0f), 1, 4);
                const uint32_t offensive_ticks = 1 + static_cast<uint32_t>(offensive_window);
                if (offensive_ticks > 1) {
                    world.schedule_event(std::make_unique<sim::OffensiveEvent>(
                                           decision.actor_country_id,
                                           target,
                                           offensive_ticks,
                                           route_distance),
                                       world.current_tick() + 1);
                }
                break;
            }
            case Strategy::Defend: {
                actor->military.units_infantry += sim::Fixed::from_double(0.35);
                actor->military.units_artillery += sim::Fixed::from_double(0.18);
                actor->military.units_air_fighter += sim::Fixed::from_double(0.08);
                actor->economic_stability += sim::Fixed::from_double(0.2);
                actor->civilian_morale += sim::Fixed::from_double(0.3);
                actor->logistics_capacity += sim::Fixed::from_double(0.25);
                actor->supply_level += sim::Fixed::from_double(0.35);
                actor->resource_reserve += sim::Fixed::from_double(0.30);
                if (actor->economic_stability > sim::Fixed::from_int(100)) {
                    actor->economic_stability = sim::Fixed::from_int(100);
                }
                if (actor->civilian_morale > sim::Fixed::from_int(100)) {
                    actor->civilian_morale = sim::Fixed::from_int(100);
                }
                if (actor->logistics_capacity > sim::Fixed::from_int(100)) {
                    actor->logistics_capacity = sim::Fixed::from_int(100);
                }
                if (actor->resource_reserve > sim::Fixed::from_int(100)) {
                    actor->resource_reserve = sim::Fixed::from_int(100);
                }
                if (actor->supply_level > sim::Fixed::from_int(100)) {
                    actor->supply_level = sim::Fixed::from_int(100);
                }
                break;
            }
            case Strategy::Negotiate: {
                if (decision.target_country_id == 0 || find_country(world, decision.target_country_id) == nullptr) {
                    return;
                }

                if (!has_reciprocal_negotiation(decision.actor_country_id, decision.target_country_id)) {
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(0.5));
                    return;
                }

                if (decision.terms.type == "alliance") {
                    bool has_target = false;
                    for (uint16_t ally_id : actor->allied_country_ids) {
                        if (ally_id == decision.target_country_id) {
                            has_target = true;
                            break;
                        }
                    }
                    if (!has_target) {
                        actor->allied_country_ids.push_back(decision.target_country_id);
                    }
                    sim::Country* target = find_country(world, decision.target_country_id);
                    if (target != nullptr) {
                        bool has_actor = false;
                        for (uint16_t ally_id : target->allied_country_ids) {
                            if (ally_id == decision.actor_country_id) {
                                has_actor = true;
                                break;
                            }
                        }
                        if (!has_actor) {
                            target->allied_country_ids.push_back(decision.actor_country_id);
                        }
                    }
                } else if (decision.terms.type == "betray") {
                    actor->allied_country_ids.erase(
                        std::remove(actor->allied_country_ids.begin(), actor->allied_country_ids.end(), decision.target_country_id),
                        actor->allied_country_ids.end());
                    sim::Country* target = find_country(world, decision.target_country_id);
                    if (target != nullptr) {
                        target->allied_country_ids.erase(
                            std::remove(target->allied_country_ids.begin(), target->allied_country_ids.end(), decision.actor_country_id),
                            target->allied_country_ids.end());
                    }
                }
                world.schedule_event(std::make_unique<sim::NegotiationEvent>(decision.actor_country_id, decision.target_country_id),
                                     world.current_tick());
                break;
            }
            case Strategy::Surrender: {
                actor->diplomatic_stance = sim::DiplomaticStance::Pacifist;
                actor->military.units_infantry = actor->military.units_infantry / sim::Fixed::from_int(2);
                actor->military.units_armor = actor->military.units_armor / sim::Fixed::from_int(2);
                actor->military.units_artillery = actor->military.units_artillery / sim::Fixed::from_int(2);
                actor->military.units_air_fighter = actor->military.units_air_fighter / sim::Fixed::from_int(2);
                actor->military.units_air_bomber = actor->military.units_air_bomber / sim::Fixed::from_int(2);
                actor->civilian_morale = sim::Fixed::from_double(10.0);
                actor->economic_stability = actor->economic_stability / sim::Fixed::from_int(2);
                if (decision.target_country_id != 0 && find_country(world, decision.target_country_id) != nullptr) {
                    world.schedule_event(std::make_unique<sim::NegotiationEvent>(decision.actor_country_id, decision.target_country_id),
                                         world.current_tick());
                }
                break;
            }
            case Strategy::TransferWeapons: {
                const uint16_t target_id = decision.target_country_id;
                if (target_id == 0 || target_id == decision.actor_country_id) {
                    return;
                }
                sim::Country* target = find_country(world, target_id);
                if (target == nullptr) {
                    return;
                }

                const sim::Fixed reserve_threshold = sim::Fixed::from_int(25);
                if (actor->resource_reserve <= reserve_threshold) {
                    return;
                }

                const sim::Fixed transfer_share = sim::Fixed::from_double(0.06 + decision.force_commitment * 0.14);
                const sim::Fixed max_bomber_transfer = sim::Fixed::from_double(3.0);
                const sim::Fixed max_surface_transfer = sim::Fixed::from_double(4.0);

                sim::Fixed bomber_transfer = actor->military.units_air_bomber * transfer_share;
                sim::Fixed surface_transfer = actor->military.units_naval_surface * transfer_share;
                if (bomber_transfer > max_bomber_transfer) {
                    bomber_transfer = max_bomber_transfer;
                }
                if (surface_transfer > max_surface_transfer) {
                    surface_transfer = max_surface_transfer;
                }

                actor->military.units_air_bomber = std::max(sim::Fixed::from_int(0), actor->military.units_air_bomber - bomber_transfer);
                actor->military.units_naval_surface = std::max(sim::Fixed::from_int(0), actor->military.units_naval_surface - surface_transfer);
                actor->resource_reserve = std::max(sim::Fixed::from_int(0), actor->resource_reserve - sim::Fixed::from_double(3.0));

                target->military.units_air_bomber += bomber_transfer;
                target->military.units_naval_surface += surface_transfer;
                target->resource_reserve = std::min(sim::Fixed::from_int(100), target->resource_reserve + sim::Fixed::from_double(2.5));
                target->civilian_morale = std::min(sim::Fixed::from_int(100), target->civilian_morale + sim::Fixed::from_double(0.8));
                actor->intelligence_level = std::min(sim::Fixed::from_int(100), actor->intelligence_level + sim::Fixed::from_double(0.4));
                break;
            }
            case Strategy::FocusEconomy: {
                actor->economic_stability = std::min(sim::Fixed::from_int(100), actor->economic_stability + sim::Fixed::from_double(2.0));
                actor->industrial_output = std::min(sim::Fixed::from_int(100), actor->industrial_output + sim::Fixed::from_double(1.5));
                actor->resource_reserve = std::min(sim::Fixed::from_int(100), actor->resource_reserve + sim::Fixed::from_double(1.3));
                actor->politics.government_stability = std::min(sim::Fixed::from_int(100), actor->politics.government_stability + sim::Fixed::from_double(1.1));
                actor->politics.public_dissent = std::max(sim::Fixed::from_int(0), actor->politics.public_dissent - sim::Fixed::from_double(0.8));
                break;
            }
            case Strategy::DevelopTechnology: {
                actor->technology_level = std::min(sim::Fixed::from_int(100), actor->technology_level + sim::Fixed::from_double(2.1));
                actor->technology.missile_defense = std::min(sim::Fixed::from_int(100), actor->technology.missile_defense + sim::Fixed::from_double(1.3));
                actor->technology.cyber_warfare = std::min(sim::Fixed::from_int(100), actor->technology.cyber_warfare + sim::Fixed::from_double(1.5));
                actor->technology.drone_operations = std::min(sim::Fixed::from_int(100), actor->technology.drone_operations + sim::Fixed::from_double(1.0));
                actor->resource_reserve = std::max(sim::Fixed::from_int(0), actor->resource_reserve - sim::Fixed::from_double(1.2));
                break;
            }
            case Strategy::FormAlliance: {
                if (decision.target_country_id == 0 || find_country(world, decision.target_country_id) == nullptr) {
                    return;
                }
                world.schedule_event(std::make_unique<sim::NegotiationEvent>(decision.actor_country_id, decision.target_country_id),
                                     world.current_tick());
                break;
            }
            case Strategy::Betray: {
                if (decision.target_country_id == 0 || find_country(world, decision.target_country_id) == nullptr) {
                    return;
                }
                actor->betrayal_tick_log.push_back(world.current_tick());
                actor->allied_country_ids.erase(
                    std::remove(actor->allied_country_ids.begin(), actor->allied_country_ids.end(), decision.target_country_id),
                    actor->allied_country_ids.end());
                sim::Country* target = find_country(world, decision.target_country_id);
                if (target != nullptr) {
                    target->allied_country_ids.erase(
                        std::remove(target->allied_country_ids.begin(), target->allied_country_ids.end(), decision.actor_country_id),
                        target->allied_country_ids.end());
                    target->trust_scores[decision.actor_country_id] = std::max(sim::Fixed::from_int(0), target->trust_scores[decision.actor_country_id] - sim::Fixed::from_double(20.0));
                }
                actor->trust_scores[decision.target_country_id] = std::max(sim::Fixed::from_int(0), actor->trust_scores[decision.target_country_id] - sim::Fixed::from_double(15.0));
                break;
            }
            case Strategy::CyberOperation: {
                if (decision.target_country_id == 0 || find_country(world, decision.target_country_id) == nullptr) {
                    return;
                }
                sim::Country* target = find_country(world, decision.target_country_id);
                if (target == nullptr) {
                    return;
                }
                const sim::Fixed cyber_adv = (actor->technology.cyber_warfare / sim::Fixed::from_int(12)) + sim::Fixed::from_double(decision.force_commitment * 1.8f);
                target->logistics_capacity = std::max(sim::Fixed::from_int(0), target->logistics_capacity - cyber_adv);
                target->intelligence_level = std::max(sim::Fixed::from_int(0), target->intelligence_level - cyber_adv / sim::Fixed::from_int(2));
                target->supply_stockpile = std::max(sim::Fixed::from_int(0), target->supply_stockpile - cyber_adv / sim::Fixed::from_int(2));
                actor->intelligence_level = std::min(sim::Fixed::from_int(100), actor->intelligence_level + sim::Fixed::from_double(0.7));
                break;
            }
            case Strategy::SignTradeAgreement: {
                const uint16_t target_id = decision.target_country_id;
                if (target_id == 0 || target_id == decision.actor_country_id) {
                    return;
                }
                sim::Country* target = find_country(world, target_id);
                if (target == nullptr) {
                    return;
                }
                if (contains_id(actor->embargoed_country_ids, target_id) || contains_id(target->embargoed_country_ids, actor->id)) {
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(0.4));
                    return;
                }
                if (has_reciprocal_trade(decision.actor_country_id, target_id) || contains_id(target->trade_partners, actor->id)) {
                    add_unique_id(&actor->trade_partners, target_id);
                    add_unique_id(&target->trade_partners, actor->id);
                    actor->trade_balance = std::min(sim::Fixed::from_int(100), actor->trade_balance + sim::Fixed::from_double(2.0));
                    target->trade_balance = std::min(sim::Fixed::from_int(100), target->trade_balance + sim::Fixed::from_double(1.4));
                    actor->economic_stability = std::min(sim::Fixed::from_int(100), actor->economic_stability + sim::Fixed::from_double(1.2));
                    target->economic_stability = std::min(sim::Fixed::from_int(100), target->economic_stability + sim::Fixed::from_double(1.2));
                    actor->trust_scores[target_id] = std::min(sim::Fixed::from_int(100), actor->trust_scores[target_id] + sim::Fixed::from_double(3.0));
                    target->trust_scores[actor->id] = std::min(sim::Fixed::from_int(100), target->trust_scores[actor->id] + sim::Fixed::from_double(3.0));
                }
                break;
            }
            case Strategy::CancelTradeAgreement: {
                const uint16_t target_id = decision.target_country_id;
                if (target_id == 0) {
                    return;
                }
                sim::Country* target = find_country(world, target_id);
                erase_id(&actor->trade_partners, target_id);
                actor->trade_balance = std::max(sim::Fixed::from_int(-100), actor->trade_balance - sim::Fixed::from_double(1.5));
                actor->economic_stability = std::max(sim::Fixed::from_int(0), actor->economic_stability - sim::Fixed::from_double(0.8));
                if (target != nullptr) {
                    erase_id(&target->trade_partners, actor->id);
                    target->economic_stability = std::max(sim::Fixed::from_int(0), target->economic_stability - sim::Fixed::from_double(0.6));
                }
                break;
            }
            case Strategy::ImposeEmbargo: {
                const uint16_t target_id = decision.target_country_id;
                if (target_id == 0 || target_id == actor->id) {
                    return;
                }
                sim::Country* target = find_country(world, target_id);
                if (target == nullptr) {
                    return;
                }
                add_unique_id(&actor->embargoed_country_ids, target_id);
                erase_id(&actor->trade_partners, target_id);
                erase_id(&target->trade_partners, actor->id);
                actor->trade_balance = std::max(sim::Fixed::from_int(-100), actor->trade_balance - sim::Fixed::from_double(1.1));
                target->trade_balance = std::max(sim::Fixed::from_int(-100), target->trade_balance - sim::Fixed::from_double(2.5));
                actor->economic_stability = std::max(sim::Fixed::from_int(0), actor->economic_stability - sim::Fixed::from_double(0.5));
                target->economic_stability = std::max(sim::Fixed::from_int(0), target->economic_stability - sim::Fixed::from_double(1.6));
                actor->trust_scores[target_id] = std::max(sim::Fixed::from_int(0), actor->trust_scores[target_id] - sim::Fixed::from_double(8.0));
                target->trust_scores[actor->id] = std::max(sim::Fixed::from_int(0), target->trust_scores[actor->id] - sim::Fixed::from_double(10.0));
                break;
            }
            case Strategy::InvestInResourceExtraction: {
                const sim::Fixed budget = std::min(actor->resource_reserve, sim::Fixed::from_double(5.5));
                if (budget <= sim::Fixed::from_double(0.5)) {
                    return;
                }
                actor->resource_reserve = std::max(sim::Fixed::from_int(0), actor->resource_reserve - budget);
                actor->resource_oil_reserves = std::min(sim::Fixed::from_int(100), actor->resource_oil_reserves + budget * sim::Fixed::from_double(0.55));
                actor->resource_minerals_reserves = std::min(sim::Fixed::from_int(100), actor->resource_minerals_reserves + budget * sim::Fixed::from_double(0.60));
                actor->resource_food_reserves = std::min(sim::Fixed::from_int(100), actor->resource_food_reserves + budget * sim::Fixed::from_double(0.45));
                actor->resource_rare_earth_reserves = std::min(sim::Fixed::from_int(100), actor->resource_rare_earth_reserves + budget * sim::Fixed::from_double(0.35));
                actor->industrial_output = std::min(sim::Fixed::from_int(100), actor->industrial_output + sim::Fixed::from_double(0.9));
                actor->faction_industrial = std::min(sim::Fixed::from_int(100), actor->faction_industrial + sim::Fixed::from_double(1.2));
                break;
            }
            case Strategy::ReduceMilitaryUpkeep: {
                actor->military.units_infantry = std::max(sim::Fixed::from_int(0), actor->military.units_infantry - actor->military.units_infantry / sim::Fixed::from_int(18));
                actor->military.units_naval_surface = std::max(sim::Fixed::from_int(0), actor->military.units_naval_surface - actor->military.units_naval_surface / sim::Fixed::from_int(24));
                actor->military_upkeep = std::max(sim::Fixed::from_int(0), actor->military_upkeep - sim::Fixed::from_double(4.0));
                actor->draft_level = std::max(sim::Fixed::from_int(0), actor->draft_level - sim::Fixed::from_double(3.0));
                actor->economic_stability = std::min(sim::Fixed::from_int(100), actor->economic_stability + sim::Fixed::from_double(1.4));
                actor->war_weariness = std::max(sim::Fixed::from_int(0), actor->war_weariness - sim::Fixed::from_double(2.2));
                actor->faction_military = std::max(sim::Fixed::from_int(0), actor->faction_military - sim::Fixed::from_double(1.5));
                actor->faction_civilian = std::min(sim::Fixed::from_int(100), actor->faction_civilian + sim::Fixed::from_double(1.1));
                break;
            }
            case Strategy::SuppressDissent: {
                actor->politics.public_dissent = std::max(sim::Fixed::from_int(0), actor->politics.public_dissent - sim::Fixed::from_double(4.5));
                actor->politics.government_stability = std::max(sim::Fixed::from_int(0), actor->politics.government_stability - sim::Fixed::from_double(1.3));
                actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(1.7));
                actor->coup_risk = std::min(sim::Fixed::from_int(100), actor->coup_risk + sim::Fixed::from_double(1.0));
                actor->faction_military = std::min(sim::Fixed::from_int(100), actor->faction_military + sim::Fixed::from_double(0.8));
                break;
            }
            case Strategy::HoldElections: {
                actor->election_cycle = 8;
                actor->politics.government_stability = std::min(sim::Fixed::from_int(100), actor->politics.government_stability + sim::Fixed::from_double(3.5));
                actor->civilian_morale = std::min(sim::Fixed::from_int(100), actor->civilian_morale + sim::Fixed::from_double(2.0));
                actor->politics.public_dissent = std::max(sim::Fixed::from_int(0), actor->politics.public_dissent - sim::Fixed::from_double(2.2));
                actor->faction_civilian = std::min(sim::Fixed::from_int(100), actor->faction_civilian + sim::Fixed::from_double(2.0));
                break;
            }
            case Strategy::CoupAttempt: {
                const double chance = (actor->coup_risk.to_double() + actor->faction_military.to_double() - actor->faction_civilian.to_double()) / 140.0;
                const double roll = static_cast<double>((world.current_tick() + actor->id * 19U) % 100U) / 100.0;
                if (roll < std::clamp(chance, 0.05, 0.85)) {
                    actor->diplomatic_stance = sim::DiplomaticStance::Aggressive;
                    actor->politics.government_stability = std::min(sim::Fixed::from_int(100), actor->politics.government_stability + sim::Fixed::from_double(6.0));
                    actor->faction_military = std::min(sim::Fixed::from_int(100), actor->faction_military + sim::Fixed::from_double(8.0));
                    actor->faction_civilian = std::max(sim::Fixed::from_int(0), actor->faction_civilian - sim::Fixed::from_double(6.0));
                    actor->coup_risk = std::max(sim::Fixed::from_int(0), actor->coup_risk - sim::Fixed::from_double(12.0));
                    actor->trade_partners.clear();
                    actor->embargoed_country_ids.clear();
                } else {
                    actor->politics.government_stability = std::max(sim::Fixed::from_int(0), actor->politics.government_stability - sim::Fixed::from_double(8.0));
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(6.0));
                    actor->coup_risk = std::min(sim::Fixed::from_int(100), actor->coup_risk + sim::Fixed::from_double(10.0));
                }
                break;
            }
            case Strategy::ProposeDefensePact: {
                const uint16_t target_id = decision.target_country_id;
                sim::Country* target = find_country(world, target_id);
                if (target == nullptr || !has_reciprocal_defense(actor->id, target_id)) {
                    return;
                }
                add_unique_id(&actor->has_defense_pact_with, target_id);
                add_unique_id(&target->has_defense_pact_with, actor->id);
                actor->defense_pact_expiry_ticks[target_id] = world.current_tick() + 48;
                target->defense_pact_expiry_ticks[actor->id] = world.current_tick() + 48;
                actor->trust_scores[target_id] = std::min(sim::Fixed::from_int(100), actor->trust_scores[target_id] + sim::Fixed::from_double(4.0));
                target->trust_scores[actor->id] = std::min(sim::Fixed::from_int(100), target->trust_scores[actor->id] + sim::Fixed::from_double(4.0));
                actor->reputation = std::min(sim::Fixed::from_int(100), actor->reputation + sim::Fixed::from_double(1.2));
                break;
            }
            case Strategy::ProposeNonAggression: {
                const uint16_t target_id = decision.target_country_id;
                sim::Country* target = find_country(world, target_id);
                if (target == nullptr || !has_reciprocal_non_aggression(actor->id, target_id)) {
                    return;
                }
                add_unique_id(&actor->has_non_aggression_with, target_id);
                add_unique_id(&target->has_non_aggression_with, actor->id);
                actor->non_aggression_expiry_ticks[target_id] = world.current_tick() + 24;
                target->non_aggression_expiry_ticks[actor->id] = world.current_tick() + 24;
                actor->trust_scores[target_id] = std::min(sim::Fixed::from_int(100), actor->trust_scores[target_id] + sim::Fixed::from_double(2.5));
                target->trust_scores[actor->id] = std::min(sim::Fixed::from_int(100), target->trust_scores[actor->id] + sim::Fixed::from_double(2.5));
                break;
            }
            case Strategy::BreakTreaty: {
                const uint16_t target_id = decision.target_country_id;
                sim::Country* target = find_country(world, target_id);
                actor->betrayal_tick_log.push_back(world.current_tick());
                erase_id(&actor->has_defense_pact_with, target_id);
                erase_id(&actor->has_non_aggression_with, target_id);
                erase_id(&actor->has_trade_treaty_with, target_id);
                actor->defense_pact_expiry_ticks.erase(target_id);
                actor->non_aggression_expiry_ticks.erase(target_id);
                actor->trade_treaty_expiry_ticks.erase(target_id);
                actor->reputation = std::max(sim::Fixed::from_int(0), actor->reputation - sim::Fixed::from_double(6.0));
                if (target != nullptr) {
                    erase_id(&target->has_defense_pact_with, actor->id);
                    erase_id(&target->has_non_aggression_with, actor->id);
                    erase_id(&target->has_trade_treaty_with, actor->id);
                    target->defense_pact_expiry_ticks.erase(actor->id);
                    target->non_aggression_expiry_ticks.erase(actor->id);
                    target->trade_treaty_expiry_ticks.erase(actor->id);
                    target->trust_scores[actor->id] = std::max(sim::Fixed::from_int(0), target->trust_scores[actor->id] - sim::Fixed::from_double(12.0));
                }
                break;
            }
            case Strategy::RequestIntel: {
                const uint16_t target_id = decision.target_country_id;
                if (target_id == 0) {
                    return;
                }
                actor->resource_reserve = std::max(sim::Fixed::from_int(0), actor->resource_reserve - sim::Fixed::from_double(1.4));
                actor->intel_on_enemy[target_id] = std::min(sim::Fixed::from_int(100), actor->intel_on_enemy[target_id] + sim::Fixed::from_double(15.0));
                actor->opponent_model_confidence[target_id] = std::min(sim::Fixed::from_int(100), actor->opponent_model_confidence[target_id] + sim::Fixed::from_double(6.0));
                actor->intelligence_level = std::min(sim::Fixed::from_int(100), actor->intelligence_level + sim::Fixed::from_double(0.9));
                break;
            }
            case Strategy::DeployUnits: {
                const sim::Fixed ground_push = sim::Fixed::from_double(0.12 + decision.force_commitment * (0.55 + decision.allocation[0] * 1.1));
                const sim::Fixed air_push = sim::Fixed::from_double(0.06 + decision.force_commitment * (0.30 + decision.allocation[1] * 0.7));
                actor->military.units_infantry += ground_push;
                actor->military.units_armor += ground_push / sim::Fixed::from_int(2);
                actor->military.units_artillery += ground_push / sim::Fixed::from_int(2);
                actor->military.units_air_fighter += air_push;
                actor->supply_level = std::max(sim::Fixed::from_int(0), actor->supply_level - sim::Fixed::from_double(0.8 + decision.force_commitment * 1.2));
                actor->deterrence_posture = std::min(sim::Fixed::from_int(100), actor->deterrence_posture + sim::Fixed::from_double(1.1));
                break;
            }
            case Strategy::TacticalNuke: {
                sim::Country* target = find_country(world, decision.target_country_id);
                if (target == nullptr || actor->nuclear_readiness < sim::Fixed::from_int(45)) {
                    return;
                }
                target->military.units_infantry = std::max(sim::Fixed::from_int(0), target->military.units_infantry - sim::Fixed::from_double(12.0));
                target->military.units_armor = std::max(sim::Fixed::from_int(0), target->military.units_armor - sim::Fixed::from_double(6.0));
                target->military.units_artillery = std::max(sim::Fixed::from_int(0), target->military.units_artillery - sim::Fixed::from_double(8.0));
                target->supply_level = std::max(sim::Fixed::from_int(0), target->supply_level - sim::Fixed::from_double(16.0));
                target->civilian_morale = std::max(sim::Fixed::from_int(0), target->civilian_morale - sim::Fixed::from_double(12.0));
                actor->nuclear_readiness = std::max(sim::Fixed::from_int(0), actor->nuclear_readiness - sim::Fixed::from_double(8.0));
                actor->escalation_level = std::min(sim::Fixed::from_int(5), actor->escalation_level + sim::Fixed::from_int(1));
                target->escalation_level = std::min(sim::Fixed::from_int(5), target->escalation_level + sim::Fixed::from_int(1));
                actor->reputation = std::max(sim::Fixed::from_int(0), actor->reputation - sim::Fixed::from_double(10.0));
                if (target->second_strike_capable) {
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(8.0));
                }
                break;
            }
            case Strategy::StrategicNuke: {
                sim::Country* target = find_country(world, decision.target_country_id);
                if (target == nullptr || actor->nuclear_readiness < sim::Fixed::from_int(60)) {
                    return;
                }
                target->military.units_infantry = std::max(sim::Fixed::from_int(0), target->military.units_infantry - sim::Fixed::from_double(36.0));
                target->military.units_armor = std::max(sim::Fixed::from_int(0), target->military.units_armor - sim::Fixed::from_double(18.0));
                target->military.units_artillery = std::max(sim::Fixed::from_int(0), target->military.units_artillery - sim::Fixed::from_double(20.0));
                target->military.units_air_fighter = std::max(sim::Fixed::from_int(0), target->military.units_air_fighter - sim::Fixed::from_double(10.0));
                target->military.units_air_bomber = std::max(sim::Fixed::from_int(0), target->military.units_air_bomber - sim::Fixed::from_double(8.0));
                target->economic_stability = std::max(sim::Fixed::from_int(0), target->economic_stability - sim::Fixed::from_double(35.0));
                target->civilian_morale = std::max(sim::Fixed::from_int(0), target->civilian_morale - sim::Fixed::from_double(40.0));
                target->supply_capacity = std::max(sim::Fixed::from_int(0), target->supply_capacity - sim::Fixed::from_double(30.0));
                actor->nuclear_readiness = std::max(sim::Fixed::from_int(0), actor->nuclear_readiness - sim::Fixed::from_double(20.0));
                actor->escalation_level = sim::Fixed::from_int(5);
                target->escalation_level = sim::Fixed::from_int(5);
                actor->reputation = std::max(sim::Fixed::from_int(0), actor->reputation - sim::Fixed::from_double(25.0));
                if (target->second_strike_capable || target->nuclear_readiness > sim::Fixed::from_int(35)) {
                    actor->economic_stability = std::max(sim::Fixed::from_int(0), actor->economic_stability - sim::Fixed::from_double(20.0));
                    actor->civilian_morale = std::max(sim::Fixed::from_int(0), actor->civilian_morale - sim::Fixed::from_double(24.0));
                }
                break;
            }
            case Strategy::CyberAttack: {
                sim::Country* target = find_country(world, decision.target_country_id);
                if (target == nullptr) {
                    return;
                }
                const sim::Fixed cyber_adv = (actor->technology.cyber_warfare / sim::Fixed::from_int(9)) + sim::Fixed::from_double(decision.force_commitment * 2.2f);
                target->supply_level = std::max(sim::Fixed::from_int(0), target->supply_level - cyber_adv);
                target->supply_capacity = std::max(sim::Fixed::from_int(0), target->supply_capacity - cyber_adv / sim::Fixed::from_int(2));
                target->intelligence_level = std::max(sim::Fixed::from_int(0), target->intelligence_level - cyber_adv / sim::Fixed::from_int(2));
                target->logistics_capacity = std::max(sim::Fixed::from_int(0), target->logistics_capacity - cyber_adv / sim::Fixed::from_int(2));
                actor->intel_on_enemy[target->id] = std::min(sim::Fixed::from_int(100), actor->intel_on_enemy[target->id] + sim::Fixed::from_double(4.0));
                actor->opponent_model_confidence[target->id] = std::min(sim::Fixed::from_int(100), actor->opponent_model_confidence[target->id] + sim::Fixed::from_double(3.0));
                break;
            }
        }
        });
    }
}

std::vector<DiplomaticMessage> ModelManager::coordinate_and_message(const sim::World& world,
                                                                     std::vector<DecisionEnvelope>* decisions) const {
    std::vector<DiplomaticMessage> messages;
    if (decisions == nullptr || decisions->empty()) {
        return messages;
    }

    std::map<std::pair<std::string, std::string>, std::map<uint16_t, uint32_t>> attack_votes;
    for (const DecisionEnvelope& envelope : *decisions) {
        if (envelope.decision.strategy != Strategy::Attack || envelope.decision.target_country_id == 0) {
            continue;
        }
        attack_votes[{envelope.model_name, envelope.team}][envelope.decision.target_country_id] += 1U;
    }

    std::map<std::pair<std::string, std::string>, uint16_t> focus_target;
    for (const auto& it : attack_votes) {
        uint16_t best_target = 0;
        uint32_t best_votes = 0;
        for (const auto& vote : it.second) {
            if (vote.second > best_votes) {
                best_votes = vote.second;
                best_target = vote.first;
            }
        }
        if (best_target != 0) {
            focus_target[it.first] = best_target;
            DiplomaticMessage msg;
            msg.from_model = it.first.first;
            msg.to_model = it.first.first;
            msg.channel = "alliance";
            msg.content = "focus_attack:" + std::to_string(best_target);
            messages.push_back(std::move(msg));
        }
    }

    for (DecisionEnvelope& envelope : *decisions) {
        const auto key = std::make_pair(envelope.model_name, envelope.team);
        auto focus_it = focus_target.find(key);
        if (focus_it != focus_target.end() && envelope.decision.strategy == Strategy::Attack) {
            envelope.decision.target_country_id = focus_it->second;
        }

        if ((envelope.decision.strategy == Strategy::Negotiate || envelope.decision.strategy == Strategy::SignTradeAgreement) &&
            envelope.decision.target_country_id != 0) {
            const sim::Country* actor = find_country(world, envelope.decision.actor_country_id);
            const sim::Country* target = find_country(world, envelope.decision.target_country_id);
            if (actor != nullptr && target != nullptr) {
                const double actor_power = actor->military.weighted_total().to_double();
                const double target_power = target->military.weighted_total().to_double();
                const double actor_pressure = (100.0 - actor->economic_stability.to_double()) + (100.0 - actor->civilian_morale.to_double());
                const double target_pressure = (100.0 - target->economic_stability.to_double()) + (100.0 - target->civilian_morale.to_double());

                if (target_power > actor_power * 1.25 && actor_pressure > 45.0) {
                    envelope.decision.terms.type = "alliance";
                    envelope.decision.terms.details = "Mutual support under high pressure";
                } else if (actor_power > target_power * 1.25 && target_pressure < 25.0) {
                    envelope.decision.terms.type = "betray";
                    envelope.decision.terms.details = "Alliance no longer beneficial";
                } else if (envelope.decision.strategy == Strategy::SignTradeAgreement) {
                    envelope.decision.terms.type = "trade";
                    envelope.decision.terms.details = "Reciprocal market access and shipping lanes";
                }
            }
        }

        if ((envelope.decision.strategy == Strategy::Negotiate || envelope.decision.strategy == Strategy::SignTradeAgreement) &&
            envelope.decision.target_country_id != 0) {
            DiplomaticMessage msg;
            msg.from_model = envelope.model_name;
            msg.to_model = team_for_country(envelope.decision.target_country_id);
            msg.channel = "diplomacy";
            msg.content = envelope.decision.terms.type.empty() ? "ceasefire" : envelope.decision.terms.type;
            messages.push_back(std::move(msg));
        }
    }

    for (DecisionEnvelope& envelope : *decisions) {
        const sim::Country* actor = find_country(world, envelope.decision.actor_country_id);
        if (actor == nullptr) {
            continue;
        }
        if (actor->civilian_morale < sim::Fixed::from_int(25) || actor->economic_stability < sim::Fixed::from_int(20)) {
            envelope.decision.strategy = Strategy::Defend;
        } else if (actor->coup_risk > sim::Fixed::from_int(80) && envelope.decision.strategy == Strategy::Attack) {
            envelope.decision.strategy = Strategy::SuppressDissent;
        }
    }

    return messages;
}

void ModelManager::set_distributed_partition(uint32_t node_id, uint32_t total_nodes) {
    distributed_total_nodes_ = std::max<uint32_t>(1U, total_nodes);
    distributed_node_id_ = node_id % distributed_total_nodes_;
}

bool ModelManager::replace_model_weights(const std::string& model_name,
                                         const std::string& state_path,
                                         std::string* error_message) {
    for (ManagedModel& managed : models_) {
        if (managed.name != model_name) {
            continue;
        }
        try {
            size_t file_in = 0;
            size_t file_out = 0;
            ModelConfig file_cfg;
            std::string inspect_error;
            if (!inspect_model_state(state_path, &file_in, &file_out, &file_cfg, &inspect_error)) {
                if (error_message != nullptr) {
                    *error_message = inspect_error;
                }
                return false;
            }

            if (file_in == 0 || file_out == 0) {
                if (error_message != nullptr) {
                    *error_message = "uploaded model has invalid architecture";
                }
                return false;
            }

            if (!validate_battle_model_dims(file_in, file_out, error_message)) {
                return false;
            }

            if (managed.model) {
                const size_t current_in = managed.model->input_dim();
                const size_t current_out = managed.model->output_dim();
                if (file_in == current_in && file_out == current_out) {
                    if (!managed.model->load_state(state_path, current_in, current_out, nullptr)) {
                        if (error_message != nullptr) {
                            *error_message = "uploaded model could not be loaded with slot architecture";
                        }
                        return false;
                    }
                    managed.model->set_training(false);
                    managed.model->set_inference_only(true);
                    return true;
                }
            }

            auto replacement = std::make_shared<Model>(file_in, file_out, file_cfg);
            if (!replacement->load_state(state_path, file_in, file_out, nullptr)) {
                if (error_message != nullptr) {
                    *error_message = "uploaded model could not be loaded with inspected architecture";
                }
                return false;
            }
            replacement->set_training(false);
            replacement->set_inference_only(true);
            managed.model = std::move(replacement);
            return true;
        } catch (const std::exception& ex) {
            if (error_message != nullptr) {
                *error_message = ex.what();
            }
            return false;
        } catch (...) {
            if (error_message != nullptr) {
                *error_message = "unknown model load error";
            }
            return false;
        }
    }
    if (error_message != nullptr) {
        *error_message = "model name not found: " + model_name;
    }
    return false;
}

bool ModelManager::reset_models_to_configured(std::string* error_message) {
    for (ManagedModel& managed : models_) {
        if (managed.configured_model_path.empty()) {
            managed.model.reset();
            continue;
        }

        std::string local_error;
        std::shared_ptr<Model> restored = load_configured_model(managed.configured_model_path, &local_error);
        if (!restored) {
            managed.model.reset();
            if (error_message != nullptr) {
                *error_message = "failed to restore configured model for slot '" + managed.name + "': " + local_error;
            }
            return false;
        }
        managed.model = std::move(restored);
    }
    return true;
}

std::vector<std::string> ModelManager::model_names() const {
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const ManagedModel& managed : models_) {
        names.push_back(managed.name);
    }
    return names;
}

std::vector<std::string> ModelManager::model_slots_for_team(const std::string& team) const {
    std::vector<std::string> out;
    if (team.empty()) {
        return out;
    }
    for (const ManagedModel& managed : models_) {
        if (managed.team == team) {
            out.push_back(managed.name);
        }
    }
    return out;
}

std::string ModelManager::team_for_country(uint16_t country_id) const {
    for (const ManagedModel& managed : models_) {
        for (uint16_t controlled : managed.controlled_country_ids) {
            if (controlled == country_id) {
                return managed.team;
            }
        }
    }
    return "unassigned";
}

std::string ModelManager::team_for_model(const std::string& model_name) const {
    for (const ManagedModel& managed : models_) {
        if (managed.name == model_name) {
            return managed.team;
        }
    }
    return "unassigned";
}

std::string ModelManager::model_for_country(uint16_t country_id) const {
    for (const ManagedModel& managed : models_) {
        for (uint16_t controlled : managed.controlled_country_ids) {
            if (controlled == country_id) {
                if (!managed.model) {
                    return "unassigned";
                }
                return managed.name;
            }
        }
    }
    return "unassigned";
}

std::string ModelManager::model_slot_for_country(uint16_t country_id) const {
    for (const ManagedModel& managed : models_) {
        for (uint16_t controlled : managed.controlled_country_ids) {
            if (controlled == country_id) {
                return managed.name;
            }
        }
    }
    return "";
}

bool ModelManager::has_loaded_model_for_country(uint16_t country_id) const {
    for (const ManagedModel& managed : models_) {
        for (uint16_t controlled : managed.controlled_country_ids) {
            if (controlled == country_id) {
                return static_cast<bool>(managed.model);
            }
        }
    }
    return false;
}

WorldSnapshot ModelManager::build_world_snapshot(const sim::World& world) const {
    WorldSnapshot snapshot;
    snapshot.tick = world.current_tick();

    snapshot.countries.reserve(world.countries().size());
    for (const sim::Country& country : world.countries()) {
        CountrySnapshot country_snapshot;
        country_snapshot.id = country.id;
        country_snapshot.color = country.color;
        country_snapshot.population = country.population;
        country_snapshot.units_infantry_milli = country.military.units_infantry.raw();
        country_snapshot.units_armor_milli = country.military.units_armor.raw();
        country_snapshot.units_artillery_milli = country.military.units_artillery.raw();
        country_snapshot.units_air_fighter_milli = country.military.units_air_fighter.raw();
        country_snapshot.units_air_bomber_milli = country.military.units_air_bomber.raw();
        country_snapshot.units_naval_surface_milli = country.military.units_naval_surface.raw();
        country_snapshot.units_naval_submarine_milli = country.military.units_naval_submarine.raw();
        country_snapshot.economic_stability_milli = country.economic_stability.raw();
        country_snapshot.civilian_morale_milli = country.civilian_morale.raw();
        country_snapshot.logistics_milli = country.logistics_capacity.raw();
        country_snapshot.intelligence_milli = country.intelligence_level.raw();
        country_snapshot.industry_milli = country.industrial_output.raw();
        country_snapshot.technology_milli = country.technology_level.raw();
        country_snapshot.resource_reserve_milli = country.resource_reserve.raw();
        country_snapshot.supply_level_milli = country.supply_level.raw();
        country_snapshot.supply_capacity_milli = country.supply_capacity.raw();
        country_snapshot.trade_balance_milli = country.trade_balance.raw();
        country_snapshot.trade_partner_ids = country.trade_partners;
        country_snapshot.defense_pact_ids = country.has_defense_pact_with;
        country_snapshot.non_aggression_pact_ids = country.has_non_aggression_with;
        country_snapshot.trade_treaty_ids = country.has_trade_treaty_with;
        country_snapshot.resource_oil_reserves_milli = country.resource_oil_reserves.raw();
        country_snapshot.resource_minerals_reserves_milli = country.resource_minerals_reserves.raw();
        country_snapshot.resource_food_reserves_milli = country.resource_food_reserves.raw();
        country_snapshot.resource_rare_earth_reserves_milli = country.resource_rare_earth_reserves.raw();
        country_snapshot.military_upkeep_milli = country.military_upkeep.raw();
        country_snapshot.faction_military_milli = country.faction_military.raw();
        country_snapshot.faction_industrial_milli = country.faction_industrial.raw();
        country_snapshot.faction_civilian_milli = country.faction_civilian.raw();
        country_snapshot.coup_risk_milli = country.coup_risk.raw();
        country_snapshot.election_cycle = country.election_cycle;
        country_snapshot.draft_level_milli = country.draft_level.raw();
        country_snapshot.war_weariness_milli = country.war_weariness.raw();
        country_snapshot.weather_severity_milli = country.weather_severity.raw();
        country_snapshot.seasonal_effect_milli = country.seasonal_effect.raw();
        country_snapshot.supply_stockpile_milli = country.supply_stockpile.raw();
        country_snapshot.terrain_mountains_milli = country.terrain.mountains.raw();
        country_snapshot.terrain_forests_milli = country.terrain.forests.raw();
        country_snapshot.terrain_urban_milli = country.terrain.urban.raw();
        country_snapshot.tech_missile_defense_milli = country.technology.missile_defense.raw();
        country_snapshot.tech_cyber_warfare_milli = country.technology.cyber_warfare.raw();
        country_snapshot.tech_electronic_warfare_milli = country.technology.electronic_warfare.raw();
        country_snapshot.tech_drone_ops_milli = country.technology.drone_operations.raw();
        country_snapshot.resource_oil_milli = country.resources.oil.raw();
        country_snapshot.resource_minerals_milli = country.resources.minerals.raw();
        country_snapshot.resource_food_milli = country.resources.food.raw();
        country_snapshot.resource_rare_earth_milli = country.resources.rare_earth.raw();
        country_snapshot.gov_stability_milli = country.politics.government_stability.raw();
        country_snapshot.public_dissent_milli = country.politics.public_dissent.raw();
        country_snapshot.corruption_milli = country.politics.corruption.raw();
        country_snapshot.nuclear_readiness_milli = country.nuclear_readiness.raw();
        country_snapshot.deterrence_posture_milli = country.deterrence_posture.raw();
        country_snapshot.reputation_milli = country.reputation.raw();
        country_snapshot.escalation_level_milli = country.escalation_level.raw();
        if (!country.trust_scores.empty()) {
            int64_t trust_sum = 0;
            for (const auto& kv : country.trust_scores) {
                trust_sum += kv.second.raw();
                country_snapshot.trust_in_milli[kv.first] = kv.second.raw();
            }
            country_snapshot.trust_average_milli = trust_sum / static_cast<int64_t>(country.trust_scores.size());
        }
        for (const auto& kv : country.believed_army_size) {
            country_snapshot.believed_army_size_milli[kv.first] = kv.second.raw();
        }
        country_snapshot.recent_betrayals = static_cast<int32_t>(country.betrayal_tick_log.size());
        country_snapshot.strategic_depth_milli = country.strategic_depth.raw();
        for (const auto& kv : country.opponent_model_confidence) {
            country_snapshot.opponent_model_confidence_milli[kv.first] = kv.second.raw();
        }
        country_snapshot.diplomatic_stance = static_cast<uint8_t>(country.diplomatic_stance);
        country_snapshot.adjacent_country_ids = country.adjacent_country_ids;
        country_snapshot.allied_country_ids = country.allied_country_ids;
        country_snapshot.second_strike_capable = country.second_strike_capable;
        for (const auto& kv : country.intel_on_enemy) {
            country_snapshot.intel_on_enemy_milli[kv.first] = kv.second.raw();
        }
        snapshot.countries.push_back(std::move(country_snapshot));
    }

    return snapshot;
}

sim::Country* ModelManager::find_country(sim::World& world, uint16_t id) const {
    for (sim::Country& country : world.mutable_countries()) {
        if (country.id == id) {
            return &country;
        }
    }
    return nullptr;
}

const sim::Country* ModelManager::find_country(const sim::World& world, uint16_t id) const {
    for (const sim::Country& country : world.countries()) {
        if (country.id == id) {
            return &country;
        }
    }
    return nullptr;
}

ReplayLogger::ReplayLogger(const std::string& path) {
    open(path);
}

bool ReplayLogger::open(const std::string& path) {
    close();
    out_.open(path, std::ios::binary);
    if (!out_) {
        return false;
    }
    chunk_buffer_.clear();
    write_binary(out_, kReplayMagic);
    write_binary(out_, kReplayVersion);
    return static_cast<bool>(out_);
}

bool ReplayLogger::write_tick(const sim::World& world,
                             const ModelManager& model_manager,
                             const std::vector<DecisionEnvelope>& decisions) {
    if (!out_) {
        return false;
    }

    const uint64_t tick = world.current_tick();
    const uint32_t country_count = static_cast<uint32_t>(world.countries().size());
    const uint32_t decision_count = static_cast<uint32_t>(decisions.size());

    std::vector<char> frame;
    frame.reserve(1024 + country_count * 220 + decision_count * 96);
    auto append = [&frame](const void* src, size_t bytes) {
        const char* begin = reinterpret_cast<const char*>(src);
        frame.insert(frame.end(), begin, begin + bytes);
    };
    auto append_pod = [&append](const auto& value) {
        append(&value, sizeof(value));
    };
    auto append_string = [&append_pod, &append](const std::string& value) {
        const uint32_t len = static_cast<uint32_t>(value.size());
        append_pod(len);
        if (len > 0) {
            append(value.data(), len);
        }
    };

    append_pod(tick);
    append_pod(country_count);
    append_pod(decision_count);

    for (const sim::Country& country : world.countries()) {
        append_pod(country.id);
        append_string(country.name);
        append_string(country.color);
        append_pod(country.population);
        append_pod(country.military.units_infantry.raw());
        append_pod(country.military.units_armor.raw());
        append_pod(country.military.units_artillery.raw());
        append_pod(country.military.units_air_fighter.raw());
        append_pod(country.military.units_air_bomber.raw());
        append_pod(country.military.units_naval_surface.raw());
        append_pod(country.military.units_naval_submarine.raw());
        append_pod(country.economic_stability.raw());
        append_pod(country.civilian_morale.raw());
        append_pod(country.logistics_capacity.raw());
        append_pod(country.intelligence_level.raw());
        append_pod(country.industrial_output.raw());
        append_pod(country.technology_level.raw());
        append_pod(country.resource_reserve.raw());
        append_pod(country.supply_level.raw());
        append_pod(country.supply_capacity.raw());
        append_pod(country.trade_balance.raw());
        const uint32_t trade_partner_count = static_cast<uint32_t>(country.trade_partners.size());
        append_pod(trade_partner_count);
        for (uint16_t partner_id : country.trade_partners) {
            append_pod(partner_id);
        }
        const auto append_id_vector = [&append_pod](const std::vector<uint16_t>& values) {
            const uint32_t count = static_cast<uint32_t>(values.size());
            append_pod(count);
            for (uint16_t id : values) {
                append_pod(id);
            }
        };
        append_id_vector(country.has_defense_pact_with);
        append_id_vector(country.has_non_aggression_with);
        append_id_vector(country.has_trade_treaty_with);
        append_pod(country.resource_oil_reserves.raw());
        append_pod(country.resource_minerals_reserves.raw());
        append_pod(country.resource_food_reserves.raw());
        append_pod(country.resource_rare_earth_reserves.raw());
        append_pod(country.military_upkeep.raw());
        append_pod(country.faction_military.raw());
        append_pod(country.faction_industrial.raw());
        append_pod(country.faction_civilian.raw());
        append_pod(country.coup_risk.raw());
        append_pod(country.election_cycle);
        append_pod(country.draft_level.raw());
        append_pod(country.war_weariness.raw());
        append_pod(country.reputation.raw());
        append_pod(country.escalation_level.raw());
        append_pod(static_cast<int32_t>(country.betrayal_tick_log.size()));
        append_pod(country.strategic_depth.raw());
        const uint8_t stance = static_cast<uint8_t>(country.diplomatic_stance);
        append_pod(stance);
        const uint8_t second_strike = country.second_strike_capable ? 1 : 0;
        append_pod(second_strike);
        append_pod(country.territory_cells);
        append_string(model_manager.team_for_country(country.id));
    }

    for (const DecisionEnvelope& envelope : decisions) {
        append_string(envelope.model_name);
        append_string(envelope.team);
        append_pod(envelope.decision.actor_country_id);
        append_pod(envelope.decision.target_country_id);
        const uint8_t strategy = static_cast<uint8_t>(envelope.decision.strategy);
        append_pod(strategy);
        append_pod(envelope.decision.force_commitment);
        append_string(envelope.decision.terms.type);
        append_string(envelope.decision.terms.details);
        const uint8_t has_secondary = envelope.decision.has_secondary_action ? 1 : 0;
        append_pod(has_secondary);
        const uint8_t secondary_strategy = static_cast<uint8_t>(envelope.decision.secondary_action.strategy);
        append_pod(secondary_strategy);
        append_pod(envelope.decision.secondary_action.target_country_id);
        append_pod(envelope.decision.secondary_action.commitment);
        append_string(envelope.decision.secondary_action.terms.type);
        append_string(envelope.decision.secondary_action.terms.details);
    }

    const uint32_t frame_size = static_cast<uint32_t>(frame.size());
    write_bytes(reinterpret_cast<const char*>(&frame_size), sizeof(frame_size));
    if (!frame.empty()) {
        write_bytes(frame.data(), frame.size());
    }
    if (chunk_buffer_.size() >= chunk_target_bytes_) {
        return flush_chunk();
    }
    return true;
}

void ReplayLogger::close() {
    if (out_) {
        flush_chunk();
        out_.close();
    }
    chunk_buffer_.clear();
}

bool ReplayLogger::write_string(const std::string& value) {
    const uint32_t len = static_cast<uint32_t>(value.size());
    write_bytes(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) {
        write_bytes(value.data(), len);
    }
    return true;
}

void ReplayLogger::write_bytes(const char* data, size_t size) {
    if (!data || size == 0) {
        return;
    }
    chunk_buffer_.insert(chunk_buffer_.end(), data, data + size);
}

bool ReplayLogger::flush_chunk() {
    if (!out_) {
        return false;
    }
    if (chunk_buffer_.empty()) {
        return true;
    }

    std::vector<char> payload = chunk_buffer_;
    uint8_t codec = 0;
    if (compressed_ && chunk_buffer_.size() > 1024) {
        std::vector<char> encoded = rle_compress(chunk_buffer_);
        if (encoded.size() + 16 < chunk_buffer_.size()) {
            payload.swap(encoded);
            codec = 1;
        }
    }

    const uint32_t magic = kReplayChunkMagic;
    const uint8_t header_version = 1;
    const uint32_t raw_size = static_cast<uint32_t>(chunk_buffer_.size());
    const uint32_t payload_size = static_cast<uint32_t>(payload.size());
    write_binary(out_, magic);
    write_binary(out_, header_version);
    write_binary(out_, codec);
    write_binary(out_, raw_size);
    write_binary(out_, payload_size);
    if (payload_size > 0) {
        out_.write(payload.data(), static_cast<std::streamsize>(payload_size));
    }

    chunk_buffer_.clear();
    return static_cast<bool>(out_);
}

ReplayReader::ReplayReader(const std::string& path) {
    open(path);
}

bool ReplayReader::open(const std::string& path) {
    in_.close();
    in_.open(path, std::ios::binary);
    if (!in_) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    if (!read_binary(in_, &magic) || !read_binary(in_, &version)) {
        return false;
    }
    if (magic != kReplayMagic) {
        return false;
    }
    version_ = version;
    chunk_buffer_.clear();
    chunk_offset_ = 0;
    return version_ == 2 || version_ == 3 || version_ == 4 || version_ == 5 || version_ == 6;
}

bool ReplayReader::read_next(ReplayFrame* frame) {
    if (!in_ || frame == nullptr) {
        return false;
    }

    if (version_ == 2) {
        return read_frame_from_stream(in_, frame);
    }

    if (version_ != 3 && version_ != 4 && version_ != 5 && version_ != 6) {
        return false;
    }

    while (true) {
        if (chunk_offset_ + sizeof(uint32_t) > chunk_buffer_.size()) {
            if (!fill_next_chunk()) {
                return false;
            }
            continue;
        }

        uint32_t frame_size = 0;
        std::memcpy(&frame_size, chunk_buffer_.data() + chunk_offset_, sizeof(frame_size));
        chunk_offset_ += sizeof(frame_size);
        if (frame_size == 0) {
            continue;
        }
        if (chunk_offset_ + frame_size > chunk_buffer_.size()) {
            chunk_offset_ = chunk_buffer_.size();
            continue;
        }

        std::string frame_bytes(chunk_buffer_.data() + chunk_offset_, chunk_buffer_.data() + chunk_offset_ + frame_size);
        chunk_offset_ += frame_size;
        std::istringstream frame_stream(frame_bytes, std::ios::binary);
        return read_frame_from_stream(frame_stream, frame);
    }
}

bool ReplayReader::fill_next_chunk() {
    chunk_buffer_.clear();
    chunk_offset_ = 0;

    uint32_t magic = 0;
    uint8_t header_version = 0;
    uint8_t codec = 0;
    uint32_t raw_size = 0;
    uint32_t payload_size = 0;
    if (!read_binary(in_, &magic)) {
        return false;
    }
    if (!read_binary(in_, &header_version) ||
        !read_binary(in_, &codec) ||
        !read_binary(in_, &raw_size) ||
        !read_binary(in_, &payload_size)) {
        return false;
    }
    if (magic != kReplayChunkMagic || header_version != 1) {
        return false;
    }

    std::vector<char> payload(payload_size);
    if (payload_size > 0) {
        in_.read(payload.data(), static_cast<std::streamsize>(payload_size));
        if (!in_) {
            return false;
        }
    }

    if (codec == 0) {
        chunk_buffer_ = std::move(payload);
    } else if (codec == 1) {
        if (!rle_decompress(payload, &chunk_buffer_, raw_size)) {
            return false;
        }
    } else {
        return false;
    }
    return !chunk_buffer_.empty();
}

bool ReplayReader::read_frame_from_stream(std::istream& in, ReplayFrame* frame) {
    if (frame == nullptr) {
        return false;
    }

    ReplayFrame output;
    uint32_t country_count = 0;
    uint32_t decision_count = 0;
    auto read_string_stream = [&in](std::string* out) -> bool {
        uint32_t len = 0;
        if (!in.read(reinterpret_cast<char*>(&len), sizeof(len))) {
            return false;
        }
        out->assign(len, '\0');
        if (len > 0) {
            in.read(out->data(), static_cast<std::streamsize>(len));
        }
        return static_cast<bool>(in);
    };

    if (!in.read(reinterpret_cast<char*>(&output.tick), sizeof(output.tick))) {
        return false;
    }
    if (!in.read(reinterpret_cast<char*>(&country_count), sizeof(country_count)) ||
        !in.read(reinterpret_cast<char*>(&decision_count), sizeof(decision_count))) {
        return false;
    }

    output.countries.reserve(country_count);
    for (uint32_t i = 0; i < country_count; ++i) {
        ReplayCountryState country;
        if (!in.read(reinterpret_cast<char*>(&country.id), sizeof(country.id)) ||
            !read_string_stream(&country.name) ||
            !read_string_stream(&country.color) ||
            !in.read(reinterpret_cast<char*>(&country.population), sizeof(country.population)) ||
            !in.read(reinterpret_cast<char*>(&country.units_infantry_milli), sizeof(country.units_infantry_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_armor_milli), sizeof(country.units_armor_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_artillery_milli), sizeof(country.units_artillery_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_air_fighter_milli), sizeof(country.units_air_fighter_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_air_bomber_milli), sizeof(country.units_air_bomber_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_naval_surface_milli), sizeof(country.units_naval_surface_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.units_naval_submarine_milli), sizeof(country.units_naval_submarine_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.economic_stability_milli), sizeof(country.economic_stability_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.civilian_morale_milli), sizeof(country.civilian_morale_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.logistics_milli), sizeof(country.logistics_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.intelligence_milli), sizeof(country.intelligence_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.industry_milli), sizeof(country.industry_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.technology_milli), sizeof(country.technology_milli)) ||
            !in.read(reinterpret_cast<char*>(&country.resource_reserve_milli), sizeof(country.resource_reserve_milli))) {
            return false;
        }
        if (version_ >= 5) {
            uint32_t trade_partner_count = 0;
            uint32_t defense_pact_count = 0;
            uint32_t non_aggression_count = 0;
            uint32_t trade_treaty_count = 0;
            if (!in.read(reinterpret_cast<char*>(&country.supply_level_milli), sizeof(country.supply_level_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.supply_capacity_milli), sizeof(country.supply_capacity_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.trade_balance_milli), sizeof(country.trade_balance_milli)) ||
                !in.read(reinterpret_cast<char*>(&trade_partner_count), sizeof(trade_partner_count))) {
                return false;
            }
            country.trade_partner_ids.resize(trade_partner_count);
            for (uint32_t partner_idx = 0; partner_idx < trade_partner_count; ++partner_idx) {
                if (!in.read(reinterpret_cast<char*>(&country.trade_partner_ids[partner_idx]), sizeof(country.trade_partner_ids[partner_idx]))) {
                    return false;
                }
            }
            auto read_id_vector = [&in](std::vector<uint16_t>* values, uint32_t* count) {
                if (!in.read(reinterpret_cast<char*>(count), sizeof(*count))) {
                    return false;
                }
                values->resize(*count);
                for (uint32_t idx = 0; idx < *count; ++idx) {
                    if (!in.read(reinterpret_cast<char*>(&(*values)[idx]), sizeof((*values)[idx]))) {
                        return false;
                    }
                }
                return true;
            };
            if (!read_id_vector(&country.defense_pact_ids, &defense_pact_count) ||
                !read_id_vector(&country.non_aggression_pact_ids, &non_aggression_count) ||
                !read_id_vector(&country.trade_treaty_ids, &trade_treaty_count)) {
                return false;
            }
            if (!in.read(reinterpret_cast<char*>(&country.resource_oil_reserves_milli), sizeof(country.resource_oil_reserves_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.resource_minerals_reserves_milli), sizeof(country.resource_minerals_reserves_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.resource_food_reserves_milli), sizeof(country.resource_food_reserves_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.resource_rare_earth_reserves_milli), sizeof(country.resource_rare_earth_reserves_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.military_upkeep_milli), sizeof(country.military_upkeep_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.faction_military_milli), sizeof(country.faction_military_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.faction_industrial_milli), sizeof(country.faction_industrial_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.faction_civilian_milli), sizeof(country.faction_civilian_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.coup_risk_milli), sizeof(country.coup_risk_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.election_cycle), sizeof(country.election_cycle)) ||
                !in.read(reinterpret_cast<char*>(&country.draft_level_milli), sizeof(country.draft_level_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.war_weariness_milli), sizeof(country.war_weariness_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.reputation_milli), sizeof(country.reputation_milli)) ||
                !in.read(reinterpret_cast<char*>(&country.escalation_level_milli), sizeof(country.escalation_level_milli))) {
                return false;
            }
            if (version_ >= 6) {
                if (!in.read(reinterpret_cast<char*>(&country.recent_betrayals), sizeof(country.recent_betrayals)) ||
                    !in.read(reinterpret_cast<char*>(&country.strategic_depth_milli), sizeof(country.strategic_depth_milli))) {
                    return false;
                }
            }
        }
        if (!in.read(reinterpret_cast<char*>(&country.diplomatic_stance), sizeof(country.diplomatic_stance)) ||
            !in.read(reinterpret_cast<char*>(&country.second_strike_capable), sizeof(country.second_strike_capable)) ||
            !in.read(reinterpret_cast<char*>(&country.territory_cells), sizeof(country.territory_cells)) ||
            !read_string_stream(&country.team)) {
            return false;
        }
        output.countries.push_back(std::move(country));
    }

    output.decisions.reserve(decision_count);
    for (uint32_t i = 0; i < decision_count; ++i) {
        DecisionEnvelope envelope;
        uint8_t strategy = 0;
        uint8_t has_secondary = 0;
        uint8_t secondary_strategy = 0;
        if (!read_string_stream(&envelope.model_name) ||
            !read_string_stream(&envelope.team) ||
            !in.read(reinterpret_cast<char*>(&envelope.decision.actor_country_id), sizeof(envelope.decision.actor_country_id)) ||
            !in.read(reinterpret_cast<char*>(&envelope.decision.target_country_id), sizeof(envelope.decision.target_country_id)) ||
            !in.read(reinterpret_cast<char*>(&strategy), sizeof(strategy))) {
            return false;
        }
        if (version_ >= 6) {
            if (!in.read(reinterpret_cast<char*>(&envelope.decision.force_commitment), sizeof(envelope.decision.force_commitment)) ||
                !read_string_stream(&envelope.decision.terms.type) ||
                !read_string_stream(&envelope.decision.terms.details) ||
                !in.read(reinterpret_cast<char*>(&has_secondary), sizeof(has_secondary)) ||
                !in.read(reinterpret_cast<char*>(&secondary_strategy), sizeof(secondary_strategy)) ||
                !in.read(reinterpret_cast<char*>(&envelope.decision.secondary_action.target_country_id), sizeof(envelope.decision.secondary_action.target_country_id)) ||
                !in.read(reinterpret_cast<char*>(&envelope.decision.secondary_action.commitment), sizeof(envelope.decision.secondary_action.commitment)) ||
                !read_string_stream(&envelope.decision.secondary_action.terms.type) ||
                !read_string_stream(&envelope.decision.secondary_action.terms.details)) {
                return false;
            }
        } else {
            envelope.decision.force_commitment = 0.5f;
            if (!read_string_stream(&envelope.decision.terms.type) ||
                !read_string_stream(&envelope.decision.terms.details)) {
                return false;
            }
        }
        envelope.decision.strategy = static_cast<Strategy>(strategy);
        envelope.decision.has_secondary_action = has_secondary != 0;
        envelope.decision.secondary_action.strategy = static_cast<Strategy>(secondary_strategy);
        output.decisions.push_back(std::move(envelope));
    }

    *frame = std::move(output);
    return true;
}

bool ReplayReader::is_open() const {
    return static_cast<bool>(in_);
}

bool ReplayReader::read_string(std::string* out) {
    uint32_t len = 0;
    if (!read_binary(in_, &len)) {
        return false;
    }
    out->assign(len, '\0');
    if (len > 0) {
        in_.read(out->data(), static_cast<std::streamsize>(len));
    }
    return static_cast<bool>(in_);
}

BattleEngine::BattleEngine(sim::World world, ModelManager model_manager)
    : initial_world_(world), world_(std::move(world)), model_manager_(std::move(model_manager)), distributed_bus_(std::make_unique<DistributedDecisionBus>()) {}

BattleEngine::~BattleEngine() {
    pause();
}

void BattleEngine::set_mode(SimulationMode mode) {
    std::lock_guard<std::mutex> lock(mu_);
    mode_ = mode;
}

void BattleEngine::set_tick_rate(double ticks_per_second) {
    std::lock_guard<std::mutex> lock(mu_);
    ticks_per_second_ = std::max(0.1, ticks_per_second);
}

void BattleEngine::start() {
    if (!validate_model_readiness(nullptr)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        mode_ = SimulationMode::Continuous;
        battle_active_ = true;
        battle_start_time_ = std::chrono::steady_clock::now();
        last_battle_elapsed_sec_ = 0;
    }
    if (running_.exchange(true)) {
        return;
    }

    if (worker_.joinable()) {
        worker_.join();
    }

    worker_ = std::thread([this]() { run_loop(); });
}

void BattleEngine::pause() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        mode_ = SimulationMode::TurnBased;
    }
    stop_worker();
}

void BattleEngine::end_battle() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (battle_active_) {
            const auto now = std::chrono::steady_clock::now();
            last_battle_elapsed_sec_ = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(now - battle_start_time_).count());
        }
        battle_active_ = false;
        mode_ = SimulationMode::TurnBased;
    }
    stop_worker();
}

void BattleEngine::reset_battle() {
    stop_worker();

    std::lock_guard<std::mutex> lock(mu_);
    world_ = initial_world_;
    std::string restore_error;
    if (!model_manager_.reset_models_to_configured(&restore_error) && !restore_error.empty()) {
        log_model_load_error_locked("reset", restore_error);
    }
    uploaded_models_by_team_.clear();
    latest_decisions_.clear();
    latest_messages_.clear();
    finalist_models_.clear();
    eliminated_models_.clear();
    winner_model_.clear();
    winner_country_id_ = 0;
    winner_country_name_.clear();
    model_load_errors_.clear();
    battle_active_ = false;
    mode_ = SimulationMode::TurnBased;
    last_battle_elapsed_sec_ = 0;
}

void BattleEngine::log_model_load_error_locked(const std::string& context, const std::string& details) {
    const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    std::ostringstream oss;
    oss << now_ms << " context=" << context << " error=" << details;
    model_load_errors_.push_back(oss.str());
    if (model_load_errors_.size() > 96) {
        model_load_errors_.erase(model_load_errors_.begin(), model_load_errors_.begin() + 24);
    }
}

void BattleEngine::stop_worker() {
    running_.store(false);
    if (!worker_.joinable()) {
        return;
    }

    if (worker_.get_id() == std::this_thread::get_id()) {
        worker_.detach();
        return;
    }

    worker_.join();
}

bool BattleEngine::set_battle_duration_seconds(uint64_t seconds, std::string* error_message) {
    std::lock_guard<std::mutex> lock(mu_);
    if (seconds < min_battle_duration_sec_) {
        if (error_message != nullptr) {
            *error_message = "seconds must be >= min_duration_sec";
        }
        return false;
    }
    if (seconds > max_battle_duration_sec_) {
        if (error_message != nullptr) {
            *error_message = "seconds must be <= max_duration_sec";
        }
        return false;
    }
    target_battle_duration_sec_ = seconds;
    return true;
}

bool BattleEngine::set_battle_duration_bounds_seconds(uint64_t min_seconds,
                                                      uint64_t max_seconds,
                                                      std::string* error_message) {
    std::lock_guard<std::mutex> lock(mu_);
    if (min_seconds < 60) {
        if (error_message != nullptr) {
            *error_message = "min_duration_sec must be >= 60";
        }
        return false;
    }
    if (max_seconds < min_seconds) {
        if (error_message != nullptr) {
            *error_message = "max_duration_sec must be >= min_duration_sec";
        }
        return false;
    }
    if (max_seconds > 86'400ULL) {
        if (error_message != nullptr) {
            *error_message = "max_duration_sec must be <= 86400";
        }
        return false;
    }

    min_battle_duration_sec_ = min_seconds;
    max_battle_duration_sec_ = max_seconds;
    if (target_battle_duration_sec_ < min_battle_duration_sec_) {
        target_battle_duration_sec_ = min_battle_duration_sec_;
    }
    if (target_battle_duration_sec_ > max_battle_duration_sec_) {
        target_battle_duration_sec_ = max_battle_duration_sec_;
    }
    return true;
}

void BattleEngine::step_once() {
    if (!validate_model_readiness(nullptr)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    tick_locked();
}

bool BattleEngine::enable_replay_logging(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    replay_enabled_ = replay_logger_.open(path);
    return replay_enabled_;
}

bool BattleEngine::configure_distributed_core(const DistributedRuntimeConfig& config, std::string* error_message) {
    std::lock_guard<std::mutex> lock(mu_);
    distributed_config_ = config;
    model_manager_.set_distributed_partition(config.node_id, std::max<uint32_t>(1U, config.total_nodes));
    if (!distributed_bus_) {
        distributed_bus_ = std::make_unique<DistributedDecisionBus>();
    }
    return distributed_bus_->configure(distributed_config_, error_message);
}

bool BattleEngine::upload_model_binary(const std::string& model_name,
                                       const std::string& team_name,
                                       uint16_t country_id,
                                       const std::string& uploaded_label,
                                       const std::string& binary_payload,
                                       std::string* error_message,
                                       std::string* applied_model_name) {
    std::lock_guard<std::mutex> lock(mu_);
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories("../logs/uploads", ec);

    const std::string upload_label = !uploaded_label.empty() ? uploaded_label : (!model_name.empty() ? model_name : (!team_name.empty() ? team_name : "auto"));
    if (binary_payload.empty()) {
        if (error_message != nullptr) {
            *error_message = "empty model binary payload";
        }
        log_model_load_error_locked(upload_label, "empty model binary payload");
        return false;
    }
    if (binary_payload.size() > (16U * 1024U * 1024U)) {
        if (error_message != nullptr) {
            *error_message = "model binary exceeds 16MB upload cap";
        }
        log_model_load_error_locked(upload_label, "model binary exceeds 16MB upload cap");
        return false;
    }

    const uint64_t timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string path = "../logs/uploads/" + upload_label + "_" + std::to_string(timestamp) + ".bin";

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error_message != nullptr) {
            *error_message = "failed to write uploaded model file";
        }
        log_model_load_error_locked(upload_label, "failed to write uploaded model file: " + path);
        return false;
    }
    out.write(binary_payload.data(), static_cast<std::streamsize>(binary_payload.size()));
    out.close();

    std::vector<std::string> requested_slots;
    if (country_id != 0) {
        const std::string requested = model_manager_.model_slot_for_country(country_id);
        if (requested.empty()) {
            if (error_message != nullptr) {
                *error_message = "no model slot found for country_id: " + std::to_string(country_id);
            }
            log_model_load_error_locked(upload_label, "no model slot found for country_id=" + std::to_string(country_id));
            return false;
        }
        requested_slots.push_back(requested);
    } else if (!model_name.empty()) {
        requested_slots.push_back(model_name);
    } else if (!team_name.empty()) {
        requested_slots = model_manager_.model_slots_for_team(team_name);
        if (requested_slots.empty()) {
            if (error_message != nullptr) {
                *error_message = "no model slot found for team: " + team_name;
            }
            log_model_load_error_locked(upload_label, "no model slot found for team=" + team_name);
            return false;
        }
    }
    if (requested_slots.empty()) {
        if (error_message != nullptr) {
            *error_message = "at least one upload target is required: name, team, or country_id";
        }
        log_model_load_error_locked(upload_label, "missing upload target: name/team/country_id");
        return false;
    }

    std::vector<std::string> applied_slots;
    applied_slots.reserve(requested_slots.size());
    for (const std::string& requested : requested_slots) {
        std::string local_error;
        if (!model_manager_.replace_model_weights(requested, path, &local_error)) {
            if (error_message != nullptr) {
                *error_message = local_error.empty() ? "failed to apply uploaded model" : local_error;
            }
            log_model_load_error_locked(upload_label, "slot=" + requested + " error=" + (local_error.empty() ? "unknown" : local_error));
            return false;
        }

        const std::string applied_team = model_manager_.team_for_model(requested);
        const std::string key = applied_team.empty() ? requested : applied_team;
        uploaded_models_by_team_[key].push_back(upload_label);
        applied_slots.push_back(requested);
    }

    if (applied_model_name != nullptr) {
        std::ostringstream applied;
        for (size_t i = 0; i < applied_slots.size(); ++i) {
            if (i > 0) {
                applied << ',';
            }
            applied << applied_slots[i];
        }
        *applied_model_name = applied.str();
    }

    if (model_load_errors_.size() > 64) {
        model_load_errors_.erase(model_load_errors_.begin(), model_load_errors_.begin() + 16);
    }
    return true;
}

bool BattleEngine::apply_manual_override(const ManualOverrideCommand& command, std::string* error_message) {
    std::lock_guard<std::mutex> lock(mu_);
    bool actor_exists = false;
    bool target_exists = false;
    for (const sim::Country& country : world_.countries()) {
        if (country.id == command.actor_country_id) {
            actor_exists = true;
        }
        if (country.id == command.target_country_id) {
            target_exists = true;
        }
    }

    if (!actor_exists) {
        if (error_message != nullptr) {
            *error_message = "actor_country_id not found";
        }
        return false;
    }

    const bool requires_target =
        command.strategy == Strategy::Attack ||
        command.strategy == Strategy::Negotiate ||
        command.strategy == Strategy::TransferWeapons ||
        command.strategy == Strategy::FormAlliance ||
        command.strategy == Strategy::Betray ||
        command.strategy == Strategy::CyberOperation ||
        command.strategy == Strategy::SignTradeAgreement ||
        command.strategy == Strategy::CancelTradeAgreement ||
        command.strategy == Strategy::ImposeEmbargo ||
        command.strategy == Strategy::ProposeDefensePact ||
        command.strategy == Strategy::ProposeNonAggression ||
        command.strategy == Strategy::BreakTreaty ||
        command.strategy == Strategy::RequestIntel ||
        command.strategy == Strategy::DeployUnits ||
        command.strategy == Strategy::TacticalNuke ||
        command.strategy == Strategy::StrategicNuke ||
        command.strategy == Strategy::CyberAttack;

    if (requires_target && (!target_exists || command.target_country_id == 0)) {
        if (error_message != nullptr) {
            *error_message = "target_country_id is required for this strategy";
        }
        return false;
    }

    DecisionEnvelope envelope;
    envelope.model_name = "manual_override";
    envelope.team = "debug";
    envelope.decision.actor_country_id = command.actor_country_id;
    envelope.decision.target_country_id = command.target_country_id;
    envelope.decision.strategy = command.strategy;
    envelope.decision.terms.type = command.terms_type;
    envelope.decision.terms.details = command.terms_details;

    std::vector<DecisionEnvelope> one{envelope};
    model_manager_.apply_decisions(world_, one);
    latest_messages_.push_back({"manual_override", "runtime", "command_center", "applied:" + std::to_string(command.actor_country_id)});
    latest_decisions_.insert(latest_decisions_.begin(), envelope);
    if (latest_decisions_.size() > 48) {
        latest_decisions_.resize(48);
    }
    world_.run_tick();
    update_competition_state_locked();

    if (replay_enabled_) {
        replay_logger_.write_tick(world_, model_manager_, latest_decisions_);
    }
    return true;
}

bool BattleEngine::validate_model_readiness(std::string* error_message) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<uint16_t> missing;
    for (const sim::Country& country : world_.countries()) {
        if (!model_manager_.has_loaded_model_for_country(country.id)) {
            missing.push_back(country.id);
        }
    }

    if (!missing.empty()) {
        if (error_message != nullptr) {
            std::ostringstream oss;
            oss << "battle blocked: every country must have a loaded model (.bin). missing country ids: ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i > 0) {
                    oss << ',';
                }
                oss << missing[i];
            }
            *error_message = oss.str();
        }
        return false;
    }
    return true;
}

std::string BattleEngine::available_models_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    const std::vector<std::string> names = model_manager_.model_names();
    std::map<std::string, std::vector<std::string>> slots_by_team;
    for (const std::string& name : names) {
        slots_by_team[model_manager_.team_for_model(name)].push_back(name);
    }

    std::ostringstream oss;
    oss << "{";
    oss << "\"models\":[";
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << "\"" << json_escape(names[i]) << "\"";
    }
    oss << "],";

    std::vector<uint16_t> missing_country_ids;
    oss << "\"country_slots\":[";
    for (size_t i = 0; i < world_.countries().size(); ++i) {
        const sim::Country& country = world_.countries()[i];
        if (i > 0) {
            oss << ',';
        }
        const bool loaded = model_manager_.has_loaded_model_for_country(country.id);
        if (!loaded) {
            missing_country_ids.push_back(country.id);
        }
        oss << "{";
        oss << "\"country_id\":" << country.id << ',';
        oss << "\"country_name\":\"" << json_escape(country.name) << "\",";
        oss << "\"team\":\"" << json_escape(model_manager_.team_for_country(country.id)) << "\",";
        oss << "\"slot_model\":\"" << json_escape(model_manager_.model_slot_for_country(country.id)) << "\",";
        oss << "\"selected_model\":\"" << json_escape(model_manager_.model_for_country(country.id)) << "\",";
        oss << "\"loaded\":" << (loaded ? "true" : "false");
        oss << "}";
    }
    oss << "],";

    oss << "\"team_slots\":{";
    bool first_team = true;
    for (const auto& kv : slots_by_team) {
        if (!first_team) {
            oss << ',';
        }
        first_team = false;
        oss << "\"" << json_escape(kv.first) << "\":[";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            if (i > 0) {
                oss << ',';
            }
            oss << "\"" << json_escape(kv.second[i]) << "\"";
        }
        oss << "]";
    }
    oss << "},";

    oss << "\"uploaded_models\":{";
    bool first_upload_team = true;
    for (const auto& kv : uploaded_models_by_team_) {
        if (!first_upload_team) {
            oss << ',';
        }
        first_upload_team = false;
        oss << "\"" << json_escape(kv.first) << "\":[";
        std::set<std::string> unique_uploads;
        for (const std::string& name : kv.second) {
            if (!name.empty()) {
                unique_uploads.insert(name);
            }
        }

        size_t i = 0;
        for (const std::string& name : unique_uploads) {
            if (i > 0) {
                oss << ',';
            }
            oss << "\"" << json_escape(name) << "\"";
            ++i;
        }
        oss << "]";
    }
    oss << "},";

    oss << "\"readiness\":{";
    oss << "\"ready\":" << (missing_country_ids.empty() ? "true" : "false") << ',';
    oss << "\"missing_country_ids\":[";
    for (size_t i = 0; i < missing_country_ids.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << missing_country_ids[i];
    }
    oss << "]";
    oss << "}";

    oss << ",\"model_load_errors\":[";
    for (size_t i = 0; i < model_load_errors_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << "\"" << json_escape(model_load_errors_[i]) << "\"";
    }
    oss << "]";

    oss << "}";
    return oss.str();
}

ReplayFrame BattleEngine::current_frame() const {
    std::lock_guard<std::mutex> lock(mu_);
    ReplayFrame frame;
    frame.tick = world_.current_tick();

    frame.countries.reserve(world_.countries().size());
    for (const sim::Country& country : world_.countries()) {
        ReplayCountryState state;
        state.id = country.id;
        state.name = country.name;
        state.color = country.color;
        state.population = country.population;
        state.units_infantry_milli = country.military.units_infantry.raw();
        state.units_armor_milli = country.military.units_armor.raw();
        state.units_artillery_milli = country.military.units_artillery.raw();
        state.units_air_fighter_milli = country.military.units_air_fighter.raw();
        state.units_air_bomber_milli = country.military.units_air_bomber.raw();
        state.units_naval_surface_milli = country.military.units_naval_surface.raw();
        state.units_naval_submarine_milli = country.military.units_naval_submarine.raw();
        state.economic_stability_milli = country.economic_stability.raw();
        state.civilian_morale_milli = country.civilian_morale.raw();
        state.logistics_milli = country.logistics_capacity.raw();
        state.intelligence_milli = country.intelligence_level.raw();
        state.industry_milli = country.industrial_output.raw();
        state.technology_milli = country.technology_level.raw();
        state.resource_reserve_milli = country.resource_reserve.raw();
        state.supply_level_milli = country.supply_level.raw();
        state.supply_capacity_milli = country.supply_capacity.raw();
        state.trade_balance_milli = country.trade_balance.raw();
        state.trade_partner_ids = country.trade_partners;
        state.defense_pact_ids = country.has_defense_pact_with;
        state.non_aggression_pact_ids = country.has_non_aggression_with;
        state.trade_treaty_ids = country.has_trade_treaty_with;
        state.resource_oil_reserves_milli = country.resource_oil_reserves.raw();
        state.resource_minerals_reserves_milli = country.resource_minerals_reserves.raw();
        state.resource_food_reserves_milli = country.resource_food_reserves.raw();
        state.resource_rare_earth_reserves_milli = country.resource_rare_earth_reserves.raw();
        state.military_upkeep_milli = country.military_upkeep.raw();
        state.faction_military_milli = country.faction_military.raw();
        state.faction_industrial_milli = country.faction_industrial.raw();
        state.faction_civilian_milli = country.faction_civilian.raw();
        state.coup_risk_milli = country.coup_risk.raw();
        state.election_cycle = country.election_cycle;
        state.draft_level_milli = country.draft_level.raw();
        state.war_weariness_milli = country.war_weariness.raw();
        state.reputation_milli = country.reputation.raw();
        state.escalation_level_milli = country.escalation_level.raw();
        state.recent_betrayals = static_cast<int32_t>(country.betrayal_tick_log.size());
        state.strategic_depth_milli = country.strategic_depth.raw();
        state.diplomatic_stance = static_cast<uint8_t>(country.diplomatic_stance);
        state.second_strike_capable = country.second_strike_capable;
        state.territory_cells = country.territory_cells;
        state.team = model_manager_.team_for_country(country.id);
        frame.countries.push_back(std::move(state));
    }

    frame.decisions = latest_decisions_;
    return frame;
}

std::string BattleEngine::current_state_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t elapsed_sec = last_battle_elapsed_sec_;
    if (battle_active_) {
        const auto now = std::chrono::steady_clock::now();
        elapsed_sec = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now - battle_start_time_).count());
    }
    const uint64_t remaining_sec = elapsed_sec >= target_battle_duration_sec_ ? 0 : (target_battle_duration_sec_ - elapsed_sec);

    std::ostringstream oss;
    oss << "{";
    oss << "\"tick\":" << world_.current_tick() << ",";
    oss << "\"countries\":[";
    const auto& countries = world_.countries();
    for (size_t i = 0; i < countries.size(); ++i) {
        const sim::Country& c = countries[i];
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"id\":" << c.id << ',';
        oss << "\"name\":\"" << json_escape(c.name) << "\",";
        oss << "\"color\":\"" << json_escape(c.color) << "\",";
        oss << "\"team\":\"" << json_escape(model_manager_.team_for_country(c.id)) << "\",";
        oss << "\"population\":" << c.population << ',';
        oss << "\"units\":{";
        oss << "\"infantry\":" << c.military.units_infantry.raw() << ',';
        oss << "\"armor\":" << c.military.units_armor.raw() << ',';
        oss << "\"artillery\":" << c.military.units_artillery.raw() << ',';
        oss << "\"air_fighter\":" << c.military.units_air_fighter.raw() << ',';
        oss << "\"air_bomber\":" << c.military.units_air_bomber.raw() << ',';
        oss << "\"naval_surface\":" << c.military.units_naval_surface.raw() << ',';
        oss << "\"naval_submarine\":" << c.military.units_naval_submarine.raw() << "},";
        oss << "\"economic_stability\":" << c.economic_stability.raw() << ',';
        oss << "\"civilian_morale\":" << c.civilian_morale.raw() << ',';
        oss << "\"logistics_capacity\":" << c.logistics_capacity.raw() << ',';
        oss << "\"intelligence_level\":" << c.intelligence_level.raw() << ',';
        oss << "\"industrial_output\":" << c.industrial_output.raw() << ',';
        oss << "\"technology_level\":" << c.technology_level.raw() << ',';
        oss << "\"resource_reserve\":" << c.resource_reserve.raw() << ',';
        oss << "\"supply_level\":" << c.supply_level.raw() << ',';
        oss << "\"supply_capacity\":" << c.supply_capacity.raw() << ',';
        oss << "\"trade_balance\":" << c.trade_balance.raw() << ',';
        oss << "\"trade_partners\":[";
        for (size_t tp = 0; tp < c.trade_partners.size(); ++tp) {
            if (tp > 0) {
                oss << ',';
            }
            oss << c.trade_partners[tp];
        }
        oss << "],";
        auto write_id_array = [&oss](const char* key, const std::vector<uint16_t>& values) {
            oss << "\"" << key << "\":[";
            for (size_t idx = 0; idx < values.size(); ++idx) {
                if (idx > 0) {
                    oss << ',';
                }
                oss << values[idx];
            }
            oss << "],";
        };
        write_id_array("defense_pacts", c.has_defense_pact_with);
        write_id_array("non_aggression_pacts", c.has_non_aggression_with);
        write_id_array("trade_treaties", c.has_trade_treaty_with);
        oss << "\"resource_reserves\":{";
        oss << "\"oil\":" << c.resource_oil_reserves.raw() << ',';
        oss << "\"minerals\":" << c.resource_minerals_reserves.raw() << ',';
        oss << "\"food\":" << c.resource_food_reserves.raw() << ',';
        oss << "\"rare_earth\":" << c.resource_rare_earth_reserves.raw() << "},";
        oss << "\"military_upkeep\":" << c.military_upkeep.raw() << ',';
        oss << "\"factions\":{";
        oss << "\"military\":" << c.faction_military.raw() << ',';
        oss << "\"industrial\":" << c.faction_industrial.raw() << ',';
        oss << "\"civilian\":" << c.faction_civilian.raw() << "},";
        oss << "\"coup_risk\":" << c.coup_risk.raw() << ',';
        oss << "\"election_cycle\":" << c.election_cycle << ',';
        oss << "\"draft_level\":" << c.draft_level.raw() << ',';
        oss << "\"war_weariness\":" << c.war_weariness.raw() << ',';
        oss << "\"recent_betrayals\":" << c.betrayal_tick_log.size() << ',';
        oss << "\"strategic_depth\":" << c.strategic_depth.raw() << ',';
        oss << "\"weather_severity\":" << c.weather_severity.raw() << ',';
        oss << "\"seasonal_effect\":" << c.seasonal_effect.raw() << ',';
        oss << "\"supply_stockpile\":" << c.supply_stockpile.raw() << ',';
        oss << "\"terrain\":{";
        oss << "\"mountains\":" << c.terrain.mountains.raw() << ',';
        oss << "\"forests\":" << c.terrain.forests.raw() << ',';
        oss << "\"urban\":" << c.terrain.urban.raw() << "},";
        oss << "\"technology_tree\":{";
        oss << "\"missile_defense\":" << c.technology.missile_defense.raw() << ',';
        oss << "\"cyber_warfare\":" << c.technology.cyber_warfare.raw() << ',';
        oss << "\"electronic_warfare\":" << c.technology.electronic_warfare.raw() << ',';
        oss << "\"drone_operations\":" << c.technology.drone_operations.raw() << "},";
        oss << "\"resources\":{";
        oss << "\"oil\":" << c.resources.oil.raw() << ',';
        oss << "\"minerals\":" << c.resources.minerals.raw() << ',';
        oss << "\"food\":" << c.resources.food.raw() << ',';
        oss << "\"rare_earth\":" << c.resources.rare_earth.raw() << "},";
        oss << "\"internal_politics\":{";
        oss << "\"government_stability\":" << c.politics.government_stability.raw() << ',';
        oss << "\"public_dissent\":" << c.politics.public_dissent.raw() << ',';
        oss << "\"corruption\":" << c.politics.corruption.raw() << "},";
        oss << "\"nuclear_readiness\":" << c.nuclear_readiness.raw() << ',';
        oss << "\"deterrence_posture\":" << c.deterrence_posture.raw() << ',';
        oss << "\"reputation\":" << c.reputation.raw() << ',';
        oss << "\"escalation_level\":" << c.escalation_level.raw() << ',';
        oss << "\"second_strike_capable\":" << (c.second_strike_capable ? "true" : "false") << ',';
        oss << "\"diplomatic_stance\":\"" << stance_to_string(c.diplomatic_stance) << "\",";
        oss << "\"territory_cells\":" << c.territory_cells;
        oss << "}";
    }
    oss << "],";

    oss << "\"battle\":{";
    oss << "\"active\":" << (battle_active_ ? "true" : "false") << ',';
    oss << "\"elapsed_sec\":" << elapsed_sec << ',';
    oss << "\"remaining_sec\":" << remaining_sec << ',';
    oss << "\"min_duration_sec\":" << min_battle_duration_sec_ << ',';
    oss << "\"max_duration_sec\":" << max_battle_duration_sec_ << ',';
    oss << "\"target_duration_sec\":" << target_battle_duration_sec_;
    oss << "},";

    oss << "\"messages\":[";
    for (size_t i = 0; i < latest_messages_.size(); ++i) {
        const DiplomaticMessage& m = latest_messages_[i];
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"from\":\"" << json_escape(m.from_model) << "\",";
        oss << "\"to\":\"" << json_escape(m.to_model) << "\",";
        oss << "\"channel\":\"" << json_escape(m.channel) << "\",";
        oss << "\"content\":\"" << json_escape(m.content) << "\"";
        oss << "}";
    }
    oss << "],";

    oss << "\"distributed\":{";
    oss << "\"node_id\":" << distributed_config_.node_id << ',';
    oss << "\"total_nodes\":" << std::max<uint32_t>(1U, distributed_config_.total_nodes) << ',';
    oss << "\"bind_host\":\"" << json_escape(distributed_config_.bind_host) << "\",";
    oss << "\"bind_port\":" << distributed_config_.bind_port;
    oss << "},";

    oss << "\"competition\":{";
    oss << "\"finalists\":[";
    for (size_t i = 0; i < finalist_models_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << "\"" << json_escape(finalist_models_[i]) << "\"";
    }
    oss << "],";
    oss << "\"eliminated\":[";
    size_t elim_count = 0;
    for (const auto& name : eliminated_models_) {
        if (elim_count++ > 0) {
            oss << ',';
        }
        oss << "\"" << json_escape(name) << "\"";
    }
    oss << "],";
    oss << "\"winner_model\":\"" << json_escape(winner_model_) << "\",";
    oss << "\"winner_country_id\":" << winner_country_id_ << ',';
    oss << "\"winner_country_name\":\"" << json_escape(winner_country_name_) << "\"";
    oss << "},";

    oss << "\"model_load_errors\":[";
    for (size_t i = 0; i < model_load_errors_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << "\"" << json_escape(model_load_errors_[i]) << "\"";
    }
    oss << "],";

    oss << "\"decisions\":[";
    for (size_t i = 0; i < latest_decisions_.size(); ++i) {
        const DecisionEnvelope& d = latest_decisions_[i];
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"model\":\"" << json_escape(d.model_name) << "\",";
        oss << "\"team\":\"" << json_escape(d.team) << "\",";
        oss << "\"actor_country_id\":" << d.decision.actor_country_id << ',';
        oss << "\"target_country_id\":" << d.decision.target_country_id << ',';
        oss << "\"strategy\":\"" << strategy_to_string(d.decision.strategy) << "\",";
        oss << "\"force_commitment\":" << d.decision.force_commitment << ',';
        oss << "\"terms\":{"
            << "\"type\":\"" << json_escape(d.decision.terms.type) << "\",";
        oss << "\"details\":\"" << json_escape(d.decision.terms.details) << "\"},";
        oss << "\"has_secondary_action\":" << (d.decision.has_secondary_action ? "true" : "false") << ',';
        oss << "\"secondary_action\":{";
        oss << "\"strategy\":\"" << strategy_to_string(d.decision.secondary_action.strategy) << "\",";
        oss << "\"target_country_id\":" << d.decision.secondary_action.target_country_id << ',';
        oss << "\"commitment\":" << d.decision.secondary_action.commitment << ',';
        oss << "\"terms\":{";
        oss << "\"type\":\"" << json_escape(d.decision.secondary_action.terms.type) << "\",";
        oss << "\"details\":\"" << json_escape(d.decision.secondary_action.terms.details) << "\"}}";
        oss << "}";
    }
    oss << "],";

    const sim::GridMap& map = world_.map();
    oss << "\"map\":{";
    oss << "\"width\":" << map.width() << ',';
    oss << "\"height\":" << map.height() << ',';
    oss << "\"cells\":[";
    const auto& cells = map.flattened_country_ids();
    for (size_t i = 0; i < cells.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << cells[i];
    }
    oss << "]}";
    oss << "}";
    return oss.str();
}

std::string BattleEngine::current_leaderboard_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    struct Row {
        std::string model;
        std::string team;
        double score = 0.0;
        double army = 0.0;
    };

    std::map<std::string, Row> model_rows;
    for (const std::string& name : model_manager_.model_names()) {
        Row row;
        row.model = name;
        row.team = model_manager_.team_for_model(name);
        model_rows[name] = row;
    }

    for (const sim::Country& c : world_.countries()) {
        const std::string model = model_manager_.model_for_country(c.id);
        const double score = static_cast<double>(c.territory_cells) * 10.0 +
                             c.military.weighted_total().to_double() +
                             c.economic_stability.to_double() * 4.0 +
                             c.civilian_morale.to_double() * 4.0 +
                             c.logistics_capacity.to_double() * 2.0 +
                             c.intelligence_level.to_double() * 2.0 +
                             c.industrial_output.to_double() * 3.0 +
                             c.technology_level.to_double() * 2.0 +
                             c.resource_reserve.to_double() * 2.0;
        auto it = model_rows.find(model);
        if (it == model_rows.end()) {
            Row row;
            row.model = model;
            row.team = model_manager_.team_for_country(c.id);
            it = model_rows.emplace(model, std::move(row)).first;
        }
        it->second.score += score;
        it->second.army += c.military.weighted_total().to_double();
    }

    std::vector<Row> rows;
    rows.reserve(model_rows.size());
    for (const auto& kv : model_rows) {
        rows.push_back(kv.second);
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (std::abs(a.score - b.score) > 1e-6) {
            return a.score > b.score;
        }
        if (std::abs(a.army - b.army) > 1e-6) {
            return a.army > b.army;
        }
        return a.model < b.model;
    });

    std::ostringstream oss;
    oss << "{";
    oss << "\"tick\":" << world_.current_tick() << ",";
    if (!rows.empty()) {
        uint16_t winner_country_id = 0;
        std::string winner_country_name;
        double winner_country_score = -1.0;

        uint16_t fallback_country_id = 0;
        std::string fallback_country_name;
        double fallback_country_score = -1.0;

        for (const sim::Country& c : world_.countries()) {
            const double country_score = static_cast<double>(c.territory_cells) * 10.0 +
                                         c.military.weighted_total().to_double() +
                                         c.economic_stability.to_double() * 4.0 +
                                         c.civilian_morale.to_double() * 4.0 +
                                         c.logistics_capacity.to_double() * 2.0 +
                                         c.intelligence_level.to_double() * 2.0 +
                                         c.industrial_output.to_double() * 3.0 +
                                         c.technology_level.to_double() * 2.0 +
                                         c.resource_reserve.to_double() * 2.0;

            if (country_score > fallback_country_score) {
                fallback_country_score = country_score;
                fallback_country_id = c.id;
                fallback_country_name = c.name;
            }

            if (model_manager_.model_for_country(c.id) != rows[0].model) {
                continue;
            }
            if (country_score > winner_country_score) {
                winner_country_score = country_score;
                winner_country_id = c.id;
                winner_country_name = c.name;
            }
        }

        if (winner_country_id == 0) {
            winner_country_id = fallback_country_id;
            winner_country_name = fallback_country_name;
        }

        oss << "\"winner_model\":\"" << json_escape(rows[0].model) << "\",";
        oss << "\"winner_team\":\"" << json_escape(rows[0].team) << "\",";
        oss << "\"winner_country_id\":" << winner_country_id << ",";
        oss << "\"winner_country_name\":\"" << json_escape(winner_country_name) << "\",";
    }
    oss << "\"leaderboard\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"rank\":" << (i + 1) << ',';
        oss << "\"model\":\"" << json_escape(rows[i].model) << "\",";
        oss << "\"team\":\"" << json_escape(rows[i].team) << "\",";
        oss << "\"score\":" << std::llround(rows[i].score);
        oss << "}";
    }
    oss << "]}";
    return oss.str();
}

std::string BattleEngine::current_diagnostics_json() const {
    std::lock_guard<std::mutex> lock(mu_);

    DistributedDecisionBus::Stats bus_stats;
    if (distributed_bus_) {
        bus_stats = distributed_bus_->snapshot();
    }

    std::ostringstream oss;
    oss << "{";
    oss << "\"tick\":" << world_.current_tick() << ',';
    oss << "\"battle_active\":" << (battle_active_ ? "true" : "false") << ',';
    oss << "\"distributed\":{";
    oss << "\"node_id\":" << distributed_config_.node_id << ',';
    oss << "\"total_nodes\":" << std::max<uint32_t>(1U, distributed_config_.total_nodes) << ',';
    oss << "\"bind_host\":\"" << json_escape(distributed_config_.bind_host) << "\",";
    oss << "\"bind_port\":" << distributed_config_.bind_port << ',';
    oss << "\"peer_count\":" << bus_stats.peer_count << ',';
    oss << "\"exchange_count\":" << bus_stats.exchange_count << ',';
    oss << "\"packets_sent\":" << bus_stats.packets_sent << ',';
    oss << "\"packets_received\":" << bus_stats.packets_received << ',';
    oss << "\"packets_dropped\":" << bus_stats.packets_dropped;
    oss << "}";
    oss << "}";
    return oss.str();
}

void BattleEngine::run_loop() {
    while (running_.load()) {
        const auto start = std::chrono::steady_clock::now();

        bool should_end = false;

        {
            std::lock_guard<std::mutex> lock(mu_);
            if (mode_ == SimulationMode::Continuous && battle_active_) {
                tick_locked();
                const auto now = std::chrono::steady_clock::now();
                const double elapsed_sec = std::chrono::duration<double>(now - battle_start_time_).count();
                last_battle_elapsed_sec_ = static_cast<uint64_t>(elapsed_sec);
                if (elapsed_sec >= static_cast<double>(target_battle_duration_sec_) && winner_country_id_ == 0) {
                    double best_score = -1.0;
                    const sim::Country* best_country = nullptr;
                    for (const sim::Country& c : world_.countries()) {
                        const double score = static_cast<double>(c.territory_cells) * 10.0 +
                                             c.military.weighted_total().to_double() +
                                             c.economic_stability.to_double() * 4.0 +
                                             c.civilian_morale.to_double() * 4.0 +
                                             c.logistics_capacity.to_double() * 2.0 +
                                             c.intelligence_level.to_double() * 2.0 +
                                             c.industrial_output.to_double() * 3.0 +
                                             c.technology_level.to_double() * 2.0 +
                                             c.resource_reserve.to_double() * 2.0;
                        if (score > best_score) {
                            best_score = score;
                            best_country = &c;
                        }
                    }
                    if (best_country != nullptr) {
                        winner_country_id_ = best_country->id;
                        winner_country_name_ = best_country->name;
                        winner_model_ = model_manager_.model_for_country(best_country->id);
                    }
                }
                if (winner_country_id_ != 0 || elapsed_sec >= static_cast<double>(target_battle_duration_sec_)) {
                    battle_active_ = false;
                    mode_ = SimulationMode::TurnBased;
                    should_end = true;
                }
            }
        }

        if (should_end) {
            running_.store(false);
            break;
        }

        double ticks = 4.0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            ticks = std::max(0.1, ticks_per_second_);
        }

        const auto target_duration = std::chrono::duration<double>(1.0 / ticks);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < target_duration) {
            std::this_thread::sleep_for(target_duration - elapsed);
        }
    }
}

void BattleEngine::tick_locked() {
    latest_decisions_ = model_manager_.gather_decisions(world_);

    if (!finalist_models_.empty()) {
        latest_decisions_.erase(
            std::remove_if(latest_decisions_.begin(), latest_decisions_.end(), [&](const DecisionEnvelope& d) {
                return std::find(finalist_models_.begin(), finalist_models_.end(), d.model_name) == finalist_models_.end();
            }),
            latest_decisions_.end());
    }

    merge_remote_decisions_locked();
    latest_messages_ = model_manager_.coordinate_and_message(world_, &latest_decisions_);

    auto heavy_step = [this]() {
        model_manager_.apply_decisions(world_, latest_decisions_);
        world_.run_tick();
    };
    if (latest_decisions_.size() >= 8 || world_.countries().size() >= 6) {
        // Offload expensive battle resolution so UI/API-facing threads spend less time in direct compute paths.
        auto task = std::async(std::launch::async, heavy_step);
        task.get();
    } else {
        heavy_step();
    }

    update_competition_state_locked();

    if (replay_enabled_) {
        replay_logger_.write_tick(world_, model_manager_, latest_decisions_);
    }
}

void BattleEngine::update_competition_state_locked() {
    finalist_models_.clear();
    eliminated_models_.clear();
    winner_model_.clear();
    winner_country_id_ = 0;
    winner_country_name_.clear();

    std::vector<uint16_t> alive_country_ids;
    alive_country_ids.reserve(world_.countries().size());
    for (const sim::Country& c : world_.countries()) {
        const std::string model_name = model_manager_.model_for_country(c.id);
        if (c.territory_cells > 0) {
            alive_country_ids.push_back(c.id);
            finalist_models_.push_back(model_name);
        } else {
            eliminated_models_.insert(model_name);
        }
    }

    if (alive_country_ids.size() == 1) {
        winner_country_id_ = alive_country_ids[0];
    }

    if (winner_country_id_ != 0) {
        for (const sim::Country& c : world_.countries()) {
            if (c.id != winner_country_id_) {
                continue;
            }
            winner_country_name_ = c.name;
            winner_model_ = model_manager_.model_for_country(c.id);
            break;
        }
    }
}

void BattleEngine::merge_remote_decisions_locked() {
    if (!distributed_bus_) {
        return;
    }
    std::vector<DecisionEnvelope> remote;
    distributed_bus_->exchange(world_.current_tick(), latest_decisions_, &remote);
    if (!remote.empty()) {
        latest_decisions_.insert(latest_decisions_.end(), remote.begin(), remote.end());
        std::unordered_set<std::string> seen;
        std::vector<DecisionEnvelope> merged;
        merged.reserve(latest_decisions_.size());
        for (const DecisionEnvelope& decision : latest_decisions_) {
            const std::string key = decision_signature(decision);
            if (seen.insert(key).second) {
                merged.push_back(decision);
            }
        }
        latest_decisions_.swap(merged);
        std::sort(latest_decisions_.begin(), latest_decisions_.end(), [](const DecisionEnvelope& a, const DecisionEnvelope& b) {
            const int pa = strategy_priority(a.decision.strategy);
            const int pb = strategy_priority(b.decision.strategy);
            if (pa == pb) {
                return a.decision.actor_country_id < b.decision.actor_country_id;
            }
            return pa < pb;
        });
    }
}

}  // namespace battle
