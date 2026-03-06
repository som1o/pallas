#ifndef SCENARIO_CONFIG_H
#define SCENARIO_CONFIG_H

#include "battle_runtime.h"
#include "simulation_engine.h"

#include <cstdint>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

struct ScenarioModelProfile {
    std::string name;
    std::string team;
    std::string model_path;
};

struct ScenarioCountryConfig {
    uint16_t id = 0;
    std::string name;
    std::string color;
    std::string team;
    std::string controller;
    uint64_t population = 0;
    int64_t army = 200;
    int64_t navy = 40;
    int64_t air_force = 30;
    int64_t missiles = 10;
    int64_t units_infantry = 220;
    int64_t units_armor = 40;
    int64_t units_artillery = 32;
    int64_t units_air_fighter = 18;
    int64_t units_air_bomber = 12;
    int64_t units_naval_surface = 16;
    int64_t units_naval_submarine = 8;
    int64_t economic_stability = 70;
    int64_t civilian_morale = 70;
    int64_t logistics_capacity = 70;
    int64_t intelligence_level = 70;
    int64_t industrial_output = 70;
    int64_t technology_level = 60;
    int64_t resource_reserve = 70;
    int64_t supply_level = 68;
    int64_t supply_capacity = 72;
    int64_t reputation = 55;
    int64_t escalation_level = 0;
    bool second_strike_capable = false;
    uint8_t diplomatic_stance = 1;
    std::vector<uint16_t> adjacent;
    std::vector<uint16_t> alliances;
    std::vector<uint16_t> defense_pacts;
    std::vector<uint16_t> non_aggression_pacts;
    std::vector<uint16_t> trade_treaties;
    std::unordered_map<uint16_t, int64_t> intel_on_enemy;
};

struct ScenarioConfig {
    uint64_t seed = 20260304ULL;
    uint64_t tick_seconds = 3600;
    uint64_t ticks_per_match = 200;
    uint32_t map_width = 36;
    uint32_t map_height = 18;
    std::vector<uint16_t> map_cells;
    std::vector<ScenarioModelProfile> models;
    std::vector<ScenarioCountryConfig> countries;
};

bool load_scenario_config(const std::string& path, ScenarioConfig* out, std::string* error);
ScenarioConfig default_scenario_config();

sim::World world_from_scenario(const ScenarioConfig& config);
battle::ModelManager model_manager_from_scenario(
    const ScenarioConfig& config,
    const std::map<std::string, std::string>& controller_overrides = {},
    uint32_t distributed_node_id = 0,
    uint32_t distributed_total_nodes = 1);

#endif
