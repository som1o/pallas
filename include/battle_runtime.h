#ifndef BATTLE_RUNTIME_H
#define BATTLE_RUNTIME_H

#include "model.h"
#include "simulation_engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace battle {

enum class SimulationMode : uint8_t {
    TurnBased = 0,
    Continuous = 1,
};

struct ManagedModel {
    std::string name;
    std::string team;
    std::shared_ptr<Model> model;
    std::vector<uint16_t> controlled_country_ids;
    std::string configured_model_path;
};

struct DecisionEnvelope {
    std::string model_name;
    std::string team;
    ModelDecision decision;
};

struct DiplomaticMessage {
    std::string from_model;
    std::string to_model;
    std::string channel;
    std::string content;
};

struct ReplayCountryState {
    uint16_t id = 0;
    std::string name;
    std::string color;
    uint64_t population = 0;
    int64_t units_infantry_milli = 0;
    int64_t units_armor_milli = 0;
    int64_t units_artillery_milli = 0;
    int64_t units_air_fighter_milli = 0;
    int64_t units_air_bomber_milli = 0;
    int64_t units_naval_surface_milli = 0;
    int64_t units_naval_submarine_milli = 0;
    int64_t economic_stability_milli = 0;
    int64_t civilian_morale_milli = 0;
    int64_t logistics_milli = 0;
    int64_t intelligence_milli = 0;
    int64_t industry_milli = 0;
    int64_t technology_milli = 0;
    int64_t resource_reserve_milli = 0;
    int64_t supply_level_milli = 0;
    int64_t supply_capacity_milli = 0;
    int64_t trade_balance_milli = 0;
    std::vector<uint16_t> trade_partner_ids;
    std::vector<uint16_t> defense_pact_ids;
    std::vector<uint16_t> non_aggression_pact_ids;
    std::vector<uint16_t> trade_treaty_ids;
    int64_t resource_oil_reserves_milli = 0;
    int64_t resource_minerals_reserves_milli = 0;
    int64_t resource_food_reserves_milli = 0;
    int64_t resource_rare_earth_reserves_milli = 0;
    int64_t military_upkeep_milli = 0;
    int64_t faction_military_milli = 0;
    int64_t faction_industrial_milli = 0;
    int64_t faction_civilian_milli = 0;
    int64_t coup_risk_milli = 0;
    int32_t election_cycle = 0;
    int64_t draft_level_milli = 0;
    int64_t war_weariness_milli = 0;
    int64_t reputation_milli = 0;
    int64_t escalation_level_milli = 0;
    int32_t recent_betrayals = 0;
    int64_t strategic_depth_milli = 0;
    uint8_t diplomatic_stance = 1;
    bool second_strike_capable = false;
    uint32_t territory_cells = 0;
    std::string team;
};

struct ReplayFrame {
    uint64_t tick = 0;
    std::vector<ReplayCountryState> countries;
    std::vector<DecisionEnvelope> decisions;
};

struct DistributedRuntimeConfig {
    uint32_t node_id = 0;
    uint32_t total_nodes = 1;
    std::string bind_host = "0.0.0.0";
    uint16_t bind_port = 19090;
    std::vector<std::string> peer_endpoints;
    uint32_t receive_timeout_ms = 40;
};

class ModelManager {
public:
    void add_model(const ManagedModel& managed_model);
    std::vector<DecisionEnvelope> gather_decisions(const sim::World& world) const;
    void apply_decisions(sim::World& world, const std::vector<DecisionEnvelope>& decisions) const;
    std::vector<DiplomaticMessage> coordinate_and_message(const sim::World& world,
                                                          std::vector<DecisionEnvelope>* decisions) const;
    void set_distributed_partition(uint32_t node_id, uint32_t total_nodes);
    bool replace_model_weights(const std::string& model_name,
                               const std::string& state_path,
                               std::string* error_message);
    bool reset_models_to_configured(std::string* error_message);
    std::vector<std::string> model_names() const;
    std::vector<std::string> model_slots_for_team(const std::string& team) const;
    std::string team_for_model(const std::string& model_name) const;
    std::string model_for_country(uint16_t country_id) const;
    std::string model_slot_for_country(uint16_t country_id) const;
    bool has_loaded_model_for_country(uint16_t country_id) const;

    std::string team_for_country(uint16_t country_id) const;

private:
    WorldSnapshot build_world_snapshot(const sim::World& world) const;
    sim::Country* find_country(sim::World& world, uint16_t id) const;
    const sim::Country* find_country(const sim::World& world, uint16_t id) const;

    std::vector<ManagedModel> models_;
    uint32_t distributed_node_id_ = 0;
    uint32_t distributed_total_nodes_ = 1;
};

class ReplayLogger {
public:
    ReplayLogger() = default;
    explicit ReplayLogger(const std::string& path);
    bool open(const std::string& path);
    bool write_tick(const sim::World& world,
                    const ModelManager& model_manager,
                    const std::vector<DecisionEnvelope>& decisions);
    void close();

private:
    bool flush_chunk();
    bool write_string(const std::string& value);
    void write_bytes(const char* data, size_t size);

    std::ofstream out_;
    std::vector<char> chunk_buffer_;
    size_t chunk_target_bytes_ = 1 << 20;
    bool compressed_ = true;
};

class ReplayReader {
public:
    ReplayReader() = default;
    explicit ReplayReader(const std::string& path);

    bool open(const std::string& path);
    bool read_next(ReplayFrame* frame);
    bool is_open() const;

private:
    bool read_string(std::string* out);
    bool fill_next_chunk();
    bool read_frame_from_stream(std::istream& in, ReplayFrame* frame);

    std::ifstream in_;
    uint32_t version_ = 0;
    std::vector<char> chunk_buffer_;
    size_t chunk_offset_ = 0;
};

struct ManualOverrideCommand {
    uint16_t actor_country_id = 0;
    uint16_t target_country_id = 0;
    Strategy strategy = Strategy::Defend;
    std::string terms_type;
    std::string terms_details;
};

class BattleEngine {
public:
    BattleEngine(sim::World world, ModelManager model_manager);
    ~BattleEngine();

    void set_mode(SimulationMode mode);
    void set_tick_rate(double ticks_per_second);

    void start();
    void pause();
    void step_once();
    void end_battle();
    void reset_battle();
    bool set_battle_duration_seconds(uint64_t seconds, std::string* error_message = nullptr);
    bool set_battle_duration_bounds_seconds(uint64_t min_seconds,
                                            uint64_t max_seconds,
                                            std::string* error_message = nullptr);

    bool enable_replay_logging(const std::string& path);
    bool configure_distributed_core(const DistributedRuntimeConfig& config, std::string* error_message);
    bool upload_model_binary(const std::string& model_name,
                             const std::string& team_name,
                             uint16_t country_id,
                             const std::string& uploaded_label,
                             const std::string& binary_payload,
                             std::string* error_message,
                             std::string* applied_model_name = nullptr);
    bool apply_manual_override(const ManualOverrideCommand& command, std::string* error_message);
    std::string available_models_json() const;
    bool validate_model_readiness(std::string* error_message) const;

    ReplayFrame current_frame() const;
    std::string current_state_json() const;
    std::string current_leaderboard_json() const;
    std::string current_diagnostics_json() const;

private:
    void stop_worker();

    void run_loop();
    void tick_locked();
    void merge_remote_decisions_locked();
    void update_competition_state_locked();
    void log_model_load_error_locked(const std::string& context, const std::string& details);

    mutable std::mutex mu_;
    sim::World initial_world_;
    sim::World world_;
    ModelManager model_manager_;
    SimulationMode mode_ = SimulationMode::TurnBased;
    double ticks_per_second_ = 4.0;
    std::vector<DecisionEnvelope> latest_decisions_;
    std::vector<DiplomaticMessage> latest_messages_;

    bool battle_active_ = false;
    uint64_t min_battle_duration_sec_ = 60;
    uint64_t max_battle_duration_sec_ = 3600;
    uint64_t target_battle_duration_sec_ = 180;
    uint64_t last_battle_elapsed_sec_ = 0;
    std::chrono::steady_clock::time_point battle_start_time_{};

    std::unique_ptr<class DistributedDecisionBus> distributed_bus_;
    DistributedRuntimeConfig distributed_config_;

    ReplayLogger replay_logger_;
    bool replay_enabled_ = false;
    std::map<std::string, std::vector<std::string>> uploaded_models_by_team_;
    std::vector<std::string> finalist_models_;
    std::set<std::string> eliminated_models_;
    std::string winner_model_;
    uint16_t winner_country_id_ = 0;
    std::string winner_country_name_;
    std::vector<std::string> model_load_errors_;

    std::atomic<bool> running_{false};
    std::thread worker_;
};

}  // namespace battle

#endif
