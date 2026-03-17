#include "scenario_config.h"

#include "model.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <set>
#include <sstream>
#include <vector>

namespace {

std::string trim(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<uint16_t> parse_u16_list(const std::string& raw, char sep) {
    std::vector<uint16_t> out;
    std::stringstream ss(raw);
    std::string part;
    while (std::getline(ss, part, sep)) {
        const std::string token = trim(part);
        if (token.empty()) {
            continue;
        }
        try {
            out.push_back(static_cast<uint16_t>(std::stoul(token)));
        } catch (...) {
        }
    }
    return out;
}

std::vector<uint8_t> parse_u8_list(const std::string& raw, char sep) {
    std::vector<uint8_t> out;
    std::stringstream ss(raw);
    std::string part;
    while (std::getline(ss, part, sep)) {
        const std::string token = trim(part);
        if (token.empty()) {
            continue;
        }
        try {
            out.push_back(static_cast<uint8_t>(std::stoul(token)));
        } catch (...) {
        }
    }
    return out;
}

void apply_legacy_force_mapping(ScenarioCountryConfig* c) {
    if (c == nullptr) {
        return;
    }
    const bool using_defaults =
        c->units_infantry == 220 && c->units_armor == 40 && c->units_artillery == 32 &&
        c->units_air_fighter == 18 && c->units_air_bomber == 12 &&
        c->units_naval_surface == 16 && c->units_naval_submarine == 8;
    if (!using_defaults) {
        return;
    }
    c->units_infantry = std::max<int64_t>(40, c->army);
    c->units_armor = std::max<int64_t>(8, c->army / 6);
    c->units_artillery = std::max<int64_t>(6, c->army / 8);
    c->units_air_fighter = std::max<int64_t>(4, (c->air_force * 3) / 5);
    c->units_air_bomber = std::max<int64_t>(3, c->air_force / 3 + c->missiles / 4);
    c->units_naval_surface = std::max<int64_t>(2, (c->navy * 7) / 10);
    c->units_naval_submarine = std::max<int64_t>(1, c->navy / 3);
}

std::shared_ptr<Model> create_model_instance(const std::string& model_path) {
    if (model_path.empty()) {
        return nullptr;
    }

    const ModelStateInspection inspection = inspect_model_state(model_path);
    if (!inspection.ok) {
        throw std::runtime_error("failed to inspect model state '" + model_path + "': " + inspection.error_message);
    }
    const size_t input_dim = inspection.input_dim;
    const size_t output_dim = inspection.output_dim;
    ModelConfig cfg = inspection.model_config;
    if (input_dim == 0 || output_dim == 0) {
        throw std::runtime_error("invalid model architecture in state file: " + model_path);
    }
    if (input_dim != battle_common::kBattleInputDim || output_dim != battle_common::kBattleOutputDim) {
        throw std::runtime_error(
            "invalid battle model architecture " + std::to_string(input_dim) + "x" +
            std::to_string(output_dim) + " in state file: " + model_path +
            " (expected " + std::to_string(battle_common::kBattleInputDim) + "x" +
            std::to_string(battle_common::kBattleOutputDim) + ")");
    }

    auto model = std::make_shared<Model>(input_dim, output_dim, cfg);
    if (!model->load_state(model_path, input_dim, output_dim, nullptr)) {
        throw std::runtime_error("failed to load model state: " + model_path);
    }
    model->set_training(false);
    model->set_inference_only(true);
    return model;
}

bool load_scenario_text(const std::string& content, ScenarioConfig* out) {
    if (out == nullptr) {
        return false;
    }

    ScenarioConfig cfg = default_scenario_config();
    std::stringstream lines(content);
    std::string line;

    while (std::getline(lines, line)) {
        std::string s = trim(line);
        if (s.empty() || s[0] == '#') {
            continue;
        }

        if (s.rfind("seed=", 0) == 0) {
            cfg.seed = static_cast<uint64_t>(std::stoull(trim(s.substr(5))));
            continue;
        }
        if (s.rfind("tick_seconds=", 0) == 0) {
            cfg.tick_seconds = static_cast<uint64_t>(std::stoull(trim(s.substr(13))));
            continue;
        }
        if (s.rfind("ticks_per_match=", 0) == 0) {
            cfg.ticks_per_match = static_cast<uint64_t>(std::stoull(trim(s.substr(16))));
            continue;
        }
        if (s.rfind("map_width=", 0) == 0) {
            cfg.map_width = static_cast<uint32_t>(std::stoul(trim(s.substr(10))));
            continue;
        }
        if (s.rfind("map_height=", 0) == 0) {
            cfg.map_height = static_cast<uint32_t>(std::stoul(trim(s.substr(11))));
            continue;
        }
        if (s.rfind("map_cells=", 0) == 0) {
            cfg.map_cells = parse_u16_list(trim(s.substr(10)), ',');
            continue;
        }
        if (s.rfind("map_cell_tags=", 0) == 0) {
            cfg.map_cell_tags = parse_u8_list(trim(s.substr(14)), ',');
            continue;
        }
        if (s.rfind("map_sea_zones=", 0) == 0) {
            cfg.map_sea_zone_ids = parse_u16_list(trim(s.substr(14)), ',');
            continue;
        }
        if (s.rfind("model=", 0) == 0) {
            std::stringstream ss(s.substr(6));
            std::string name;
            std::string team;
            std::string path;
            std::getline(ss, name, ',');
            std::getline(ss, team, ',');
            std::getline(ss, path, ',');
            ScenarioModelProfile m;
            m.name = trim(name);
            m.team = trim(team);
            m.model_path = trim(path);
            if (!m.name.empty()) {
                cfg.models.push_back(std::move(m));
            }
            continue;
        }
        if (s.rfind("country=", 0) == 0) {
            std::vector<std::string> parts;
            std::stringstream ss(s.substr(8));
            std::string part;
            while (std::getline(ss, part, ',')) {
                parts.push_back(trim(part));
            }
            if (parts.size() < 11) {
                continue;
            }

            ScenarioCountryConfig c;
            c.id = static_cast<uint16_t>(std::stoul(parts[0]));
            c.name = parts[1];
            c.team = parts[2];
            c.controller = parts[3];
            c.army = std::stoll(parts[4]);
            c.navy = std::stoll(parts[5]);
            c.air_force = std::stoll(parts[6]);
            c.missiles = std::stoll(parts[7]);
            c.economic_stability = std::stoll(parts[8]);
            c.civilian_morale = std::stoll(parts[9]);
            c.adjacent = parse_u16_list(parts[10], '|');
            if (parts.size() >= 12) {
                c.alliances = parse_u16_list(parts[11], '|');
            }
            if (parts.size() >= 13) {
                c.population = static_cast<uint64_t>(std::stoull(parts[12]));
            }
            if (parts.size() >= 14) {
                c.diplomatic_stance = static_cast<uint8_t>(std::stoul(parts[13]));
            }
            if (parts.size() >= 15) {
                c.color = parts[14];
            }
            if (parts.size() >= 16) {
                c.logistics_capacity = std::stoll(parts[15]);
            }
            if (parts.size() >= 17) {
                c.intelligence_level = std::stoll(parts[16]);
            }
            if (parts.size() >= 18) {
                c.industrial_output = std::stoll(parts[17]);
            }
            if (parts.size() >= 19) {
                c.technology_level = std::stoll(parts[18]);
            }
            if (parts.size() >= 20) {
                c.resource_reserve = std::stoll(parts[19]);
            }
            cfg.countries.push_back(std::move(c));
            continue;
        }
    }

    const size_t cell_count = static_cast<size_t>(cfg.map_width) * cfg.map_height;
    if (!cfg.map_cell_tags.empty() && cfg.map_cell_tags.size() != cell_count) {
        cfg.map_cell_tags.clear();
    }
    if (!cfg.map_sea_zone_ids.empty() && cfg.map_sea_zone_ids.size() != cell_count) {
        cfg.map_sea_zone_ids.clear();
    }

    if (!cfg.map_cells.empty() && cfg.map_cells.size() != cell_count) {
        return false;
    }
    if (!cfg.map_cell_tags.empty() && cfg.map_cell_tags.size() != cell_count) {
        return false;
    }
    if (!cfg.map_sea_zone_ids.empty() && cfg.map_sea_zone_ids.size() != cell_count) {
        return false;
    }

    *out = std::move(cfg);
    return true;
}

bool load_scenario_json(const std::string& content, ScenarioConfig* out) {
    if (out == nullptr) {
        return false;
    }

    const auto j = nlohmann::json::parse(content);
    ScenarioConfig cfg = default_scenario_config();

    if (j.contains("seed") && j["seed"].is_number_unsigned()) {
        cfg.seed = j["seed"].get<uint64_t>();
    }
    if (j.contains("tick_seconds") && j["tick_seconds"].is_number_unsigned()) {
        cfg.tick_seconds = j["tick_seconds"].get<uint64_t>();
    }
    if (j.contains("ticks_per_match") && j["ticks_per_match"].is_number_unsigned()) {
        cfg.ticks_per_match = j["ticks_per_match"].get<uint64_t>();
    }

    if (j.contains("map") && j["map"].is_object()) {
        const auto& map = j["map"];
        if (map.contains("width") && map["width"].is_number_unsigned()) {
            cfg.map_width = map["width"].get<uint32_t>();
        }
        if (map.contains("height") && map["height"].is_number_unsigned()) {
            cfg.map_height = map["height"].get<uint32_t>();
        }
        if (map.contains("cells") && map["cells"].is_array()) {
            cfg.map_cells.clear();
            for (const auto& cell : map["cells"]) {
                if (cell.is_number_unsigned()) {
                    cfg.map_cells.push_back(cell.get<uint16_t>());
                }
            }
        }
        if (map.contains("tags") && map["tags"].is_array()) {
            cfg.map_cell_tags.clear();
            for (const auto& tag : map["tags"]) {
                if (tag.is_number_unsigned()) {
                    cfg.map_cell_tags.push_back(static_cast<uint8_t>(tag.get<uint16_t>()));
                }
            }
        }
        if (map.contains("sea_zones") && map["sea_zones"].is_array()) {
            cfg.map_sea_zone_ids.clear();
            for (const auto& zone : map["sea_zones"]) {
                if (zone.is_number_unsigned()) {
                    cfg.map_sea_zone_ids.push_back(zone.get<uint16_t>());
                }
            }
        }
    }

    if (j.contains("models") && j["models"].is_array()) {
        cfg.models.clear();
        for (const auto& item : j["models"]) {
            if (!item.is_object()) {
                continue;
            }
            ScenarioModelProfile m;
            if (item.contains("name") && item["name"].is_string()) {
                m.name = item["name"].get<std::string>();
            }
            if (item.contains("team") && item["team"].is_string()) {
                m.team = item["team"].get<std::string>();
            }
            if (item.contains("model_path") && item["model_path"].is_string()) {
                m.model_path = item["model_path"].get<std::string>();
            }
            if (!m.name.empty()) {
                cfg.models.push_back(std::move(m));
            }
        }
    }

    if (j.contains("countries") && j["countries"].is_array()) {
        cfg.countries.clear();
        for (const auto& item : j["countries"]) {
            if (!item.is_object()) {
                continue;
            }
            ScenarioCountryConfig c;
            if (item.contains("id") && item["id"].is_number_unsigned()) c.id = item["id"].get<uint16_t>();
            if (item.contains("name") && item["name"].is_string()) c.name = item["name"].get<std::string>();
            if (item.contains("color") && item["color"].is_string()) c.color = item["color"].get<std::string>();
            if (item.contains("team") && item["team"].is_string()) c.team = item["team"].get<std::string>();
            if (item.contains("controller") && item["controller"].is_string()) c.controller = item["controller"].get<std::string>();
            if (item.contains("population") && item["population"].is_number_unsigned()) c.population = item["population"].get<uint64_t>();
            if (item.contains("army") && item["army"].is_number_integer()) c.army = item["army"].get<int64_t>();
            if (item.contains("navy") && item["navy"].is_number_integer()) c.navy = item["navy"].get<int64_t>();
            if (item.contains("air_force") && item["air_force"].is_number_integer()) c.air_force = item["air_force"].get<int64_t>();
            if (item.contains("missiles") && item["missiles"].is_number_integer()) c.missiles = item["missiles"].get<int64_t>();
            if (item.contains("economic_stability") && item["economic_stability"].is_number_integer()) c.economic_stability = item["economic_stability"].get<int64_t>();
            if (item.contains("civilian_morale") && item["civilian_morale"].is_number_integer()) c.civilian_morale = item["civilian_morale"].get<int64_t>();
            if (item.contains("logistics_capacity") && item["logistics_capacity"].is_number_integer()) c.logistics_capacity = item["logistics_capacity"].get<int64_t>();
            if (item.contains("intelligence_level") && item["intelligence_level"].is_number_integer()) c.intelligence_level = item["intelligence_level"].get<int64_t>();
            if (item.contains("industrial_output") && item["industrial_output"].is_number_integer()) c.industrial_output = item["industrial_output"].get<int64_t>();
            if (item.contains("technology_level") && item["technology_level"].is_number_integer()) c.technology_level = item["technology_level"].get<int64_t>();
            if (item.contains("resource_reserve") && item["resource_reserve"].is_number_integer()) c.resource_reserve = item["resource_reserve"].get<int64_t>();
            if (item.contains("units_infantry") && item["units_infantry"].is_number_integer()) c.units_infantry = item["units_infantry"].get<int64_t>();
            if (item.contains("units_armor") && item["units_armor"].is_number_integer()) c.units_armor = item["units_armor"].get<int64_t>();
            if (item.contains("units_artillery") && item["units_artillery"].is_number_integer()) c.units_artillery = item["units_artillery"].get<int64_t>();
            if (item.contains("units_air_fighter") && item["units_air_fighter"].is_number_integer()) c.units_air_fighter = item["units_air_fighter"].get<int64_t>();
            if (item.contains("units_air_bomber") && item["units_air_bomber"].is_number_integer()) c.units_air_bomber = item["units_air_bomber"].get<int64_t>();
            if (item.contains("units_naval_surface") && item["units_naval_surface"].is_number_integer()) c.units_naval_surface = item["units_naval_surface"].get<int64_t>();
            if (item.contains("units_naval_submarine") && item["units_naval_submarine"].is_number_integer()) c.units_naval_submarine = item["units_naval_submarine"].get<int64_t>();
            if (item.contains("supply_level") && item["supply_level"].is_number_integer()) c.supply_level = item["supply_level"].get<int64_t>();
            if (item.contains("supply_capacity") && item["supply_capacity"].is_number_integer()) c.supply_capacity = item["supply_capacity"].get<int64_t>();
            if (item.contains("reputation") && item["reputation"].is_number_integer()) c.reputation = item["reputation"].get<int64_t>();
            if (item.contains("escalation_level") && item["escalation_level"].is_number_integer()) c.escalation_level = item["escalation_level"].get<int64_t>();
            if (item.contains("second_strike_capable") && item["second_strike_capable"].is_boolean()) c.second_strike_capable = item["second_strike_capable"].get<bool>();
            if (item.contains("diplomatic_stance") && item["diplomatic_stance"].is_number_unsigned()) c.diplomatic_stance = item["diplomatic_stance"].get<uint8_t>();
            if (item.contains("adjacent") && item["adjacent"].is_array()) {
                for (const auto& id : item["adjacent"]) {
                    if (id.is_number_unsigned()) c.adjacent.push_back(id.get<uint16_t>());
                }
            }
            if (item.contains("alliances") && item["alliances"].is_array()) {
                for (const auto& id : item["alliances"]) {
                    if (id.is_number_unsigned()) c.alliances.push_back(id.get<uint16_t>());
                }
            }
            if (item.contains("defense_pacts") && item["defense_pacts"].is_array()) {
                for (const auto& id : item["defense_pacts"]) {
                    if (id.is_number_unsigned()) c.defense_pacts.push_back(id.get<uint16_t>());
                }
            }
            if (item.contains("non_aggression_pacts") && item["non_aggression_pacts"].is_array()) {
                for (const auto& id : item["non_aggression_pacts"]) {
                    if (id.is_number_unsigned()) c.non_aggression_pacts.push_back(id.get<uint16_t>());
                }
            }
            if (item.contains("trade_treaties") && item["trade_treaties"].is_array()) {
                for (const auto& id : item["trade_treaties"]) {
                    if (id.is_number_unsigned()) c.trade_treaties.push_back(id.get<uint16_t>());
                }
            }
            if (item.contains("intel_on_enemy") && item["intel_on_enemy"].is_object()) {
                for (auto it = item["intel_on_enemy"].begin(); it != item["intel_on_enemy"].end(); ++it) {
                    if (it.value().is_number_integer()) {
                        c.intel_on_enemy[static_cast<uint16_t>(std::stoul(it.key()))] = it.value().get<int64_t>();
                    }
                }
            }
            apply_legacy_force_mapping(&c);
            if (c.id != 0) {
                cfg.countries.push_back(std::move(c));
            }
        }
    }

    const size_t cell_count = static_cast<size_t>(cfg.map_width) * cfg.map_height;
    if (!cfg.map_cell_tags.empty() && cfg.map_cell_tags.size() != cell_count) {
        cfg.map_cell_tags.clear();
    }
    if (!cfg.map_sea_zone_ids.empty() && cfg.map_sea_zone_ids.size() != cell_count) {
        cfg.map_sea_zone_ids.clear();
    }

    if (!cfg.map_cells.empty() && cfg.map_cells.size() != cell_count) {
        return false;
    }
    if (!cfg.map_cell_tags.empty() && cfg.map_cell_tags.size() != cell_count) {
        return false;
    }
    if (!cfg.map_sea_zone_ids.empty() && cfg.map_sea_zone_ids.size() != cell_count) {
        return false;
    }

    *out = std::move(cfg);
    return true;
}

}  // namespace

ScenarioConfig default_scenario_config() {
    ScenarioConfig cfg;

    cfg.models = {
        {"aster_ai", "aster", ""},
        {"boreal_ai", "boreal", ""},
        {"crux_ai", "crux", ""},
        {"deltora_ai", "deltora", ""},
    };

    cfg.countries.clear();
    {
        ScenarioCountryConfig c;
        c.id = 1;
        c.name = "Aster";
        c.color = "#cf6a3d";
        c.team = "aster";
        c.controller = "aster_ai";
        c.population = 9'000'000;
        c.army = 260;
        c.navy = 60;
        c.air_force = 45;
        c.missiles = 20;
        c.economic_stability = 75;
        c.civilian_morale = 70;
        c.logistics_capacity = 68;
        c.intelligence_level = 66;
        c.industrial_output = 74;
        c.technology_level = 63;
        c.resource_reserve = 69;
        c.diplomatic_stance = 0;
        c.adjacent = {2};
        c.alliances = {};
        cfg.countries.push_back(std::move(c));
    }
    {
        ScenarioCountryConfig c;
        c.id = 2;
        c.name = "Boreal";
        c.color = "#3e8f7a";
        c.team = "boreal";
        c.controller = "boreal_ai";
        c.population = 8'500'000;
        c.army = 245;
        c.navy = 55;
        c.air_force = 42;
        c.missiles = 18;
        c.economic_stability = 74;
        c.civilian_morale = 69;
        c.logistics_capacity = 72;
        c.intelligence_level = 70;
        c.industrial_output = 67;
        c.technology_level = 66;
        c.resource_reserve = 73;
        c.diplomatic_stance = 1;
        c.adjacent = {1, 3};
        c.alliances = {};
        cfg.countries.push_back(std::move(c));
    }
    {
        ScenarioCountryConfig c;
        c.id = 3;
        c.name = "Crux";
        c.color = "#d3b04d";
        c.team = "crux";
        c.controller = "crux_ai";
        c.population = 9'200'000;
        c.army = 255;
        c.navy = 58;
        c.air_force = 44;
        c.missiles = 21;
        c.economic_stability = 73;
        c.civilian_morale = 67;
        c.logistics_capacity = 65;
        c.intelligence_level = 75;
        c.industrial_output = 71;
        c.technology_level = 72;
        c.resource_reserve = 62;
        c.diplomatic_stance = 0;
        c.adjacent = {2, 4};
        c.alliances = {};
        cfg.countries.push_back(std::move(c));
    }
    {
        ScenarioCountryConfig c;
        c.id = 4;
        c.name = "Deltora";
        c.color = "#5c76c9";
        c.team = "deltora";
        c.controller = "deltora_ai";
        c.population = 8'700'000;
        c.army = 238;
        c.navy = 53;
        c.air_force = 41;
        c.missiles = 17;
        c.economic_stability = 76;
        c.civilian_morale = 72;
        c.logistics_capacity = 74;
        c.intelligence_level = 68;
        c.industrial_output = 69;
        c.technology_level = 64;
        c.resource_reserve = 78;
        c.diplomatic_stance = 1;
        c.adjacent = {3};
        c.alliances = {};
        cfg.countries.push_back(std::move(c));
    }

    cfg.map_cells.assign(static_cast<size_t>(cfg.map_width) * cfg.map_height, 0);
    cfg.map_cell_tags.assign(static_cast<size_t>(cfg.map_width) * cfg.map_height, 0);
    cfg.map_sea_zone_ids.assign(static_cast<size_t>(cfg.map_width) * cfg.map_height, 0);
    for (uint32_t y = 0; y < cfg.map_height; ++y) {
        for (uint32_t x = 0; x < cfg.map_width; ++x) {
            uint16_t id = 0;
            if (x < 9) {
                id = 1;
            } else if (x < 18) {
                id = 2;
            } else if (x < 27) {
                id = 3;
            } else {
                id = 4;
            }
            const size_t idx = static_cast<size_t>(y) * cfg.map_width + x;
            cfg.map_cells[idx] = id;

            const bool central_sea_lane = (y >= 8 && y <= 9);
            if (central_sea_lane) {
                cfg.map_cells[idx] = 0;
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(sim::GridMap::kTagSea);
                cfg.map_sea_zone_ids[idx] = static_cast<uint16_t>((x < 12) ? 1 : (x < 24 ? 2 : 3));
            }
            if ((x == 11 || x == 23) && y == 8) {
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(cfg.map_cell_tags[idx] | sim::GridMap::kTagChokepointStrait | sim::GridMap::kTagStrategic);
            }
            if ((x == 17 || x == 29) && y == 9) {
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(cfg.map_cell_tags[idx] | sim::GridMap::kTagChokepointCanal | sim::GridMap::kTagStrategic);
            }

            if (!central_sea_lane && ((y == 7 || y == 10) && (x % 9 == 0))) {
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(cfg.map_cell_tags[idx] | sim::GridMap::kTagPort | sim::GridMap::kTagStrategic);
            }
            if (!central_sea_lane && (x == 8 || x == 17 || x == 26) && (y == 5 || y == 12)) {
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(cfg.map_cell_tags[idx] | sim::GridMap::kTagMountainPass | sim::GridMap::kTagStrategic);
            }
            if (!central_sea_lane && (y == 6 || y == 11) && (x == 5 || x == 14 || x == 22 || x == 31)) {
                cfg.map_cell_tags[idx] = static_cast<uint8_t>(cfg.map_cell_tags[idx] | sim::GridMap::kTagRiverCrossing | sim::GridMap::kTagStrategic);
            }
        }
    }

    return cfg;
}

bool load_scenario_config(const std::string& path, ScenarioConfig* out, std::string* error) {
    if (out == nullptr) {
        return false;
    }

    std::ifstream in(path);
    if (!in) {
        if (error != nullptr) {
            *error = "unable to open scenario: " + path;
        }
        return false;
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string content = ss.str();

    const std::string trimmed = trim(content);
    bool ok = false;
    if (!trimmed.empty() && trimmed.front() == '{') {
        try {
            ok = load_scenario_json(content, out);
        } catch (...) {
            ok = false;
        }
    } else {
        try {
            ok = load_scenario_text(content, out);
        } catch (...) {
            ok = false;
        }
    }

    if (!ok && error != nullptr) {
    *error = "failed to parse scenario file";
    }
    return ok;
}

sim::World world_from_scenario(const ScenarioConfig& config) {
    sim::World world(config.seed, config.tick_seconds);
    world.reserve_countries(config.countries.size());

    sim::GridMap map(config.map_width, config.map_height);
    if (config.map_cells.size() == static_cast<size_t>(config.map_width) * config.map_height) {
        for (uint32_t y = 0; y < config.map_height; ++y) {
            for (uint32_t x = 0; x < config.map_width; ++x) {
                const size_t idx = static_cast<size_t>(y) * config.map_width + x;
                map.set(x, y, config.map_cells[idx]);
                if (config.map_cell_tags.size() == config.map_cells.size()) {
                    map.set_cell_tags(x, y, config.map_cell_tags[idx]);
                }
                if (config.map_sea_zone_ids.size() == config.map_cells.size()) {
                    map.set_sea_zone(x, y, config.map_sea_zone_ids[idx]);
                }
            }
        }
    }
    world.set_map(map);

    for (const ScenarioCountryConfig& c : config.countries) {
        sim::Country country;
        country.id = c.id;
        country.name = c.name;
        country.color = c.color;
        country.population = c.population == 0 ? (8'000'000 + static_cast<uint64_t>(c.id) * 1'000'000) : c.population;
        country.capital = {static_cast<int32_t>(c.id * 5), static_cast<int32_t>(c.id * 2)};
        country.shape = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        country.adjacent_country_ids = c.adjacent;
        country.allied_country_ids = c.alliances;
        ScenarioCountryConfig mapped = c;
        apply_legacy_force_mapping(&mapped);
        country.military.units_infantry = sim::Fixed::from_int(mapped.units_infantry);
        country.military.units_armor = sim::Fixed::from_int(mapped.units_armor);
        country.military.units_artillery = sim::Fixed::from_int(mapped.units_artillery);
        country.military.units_air_fighter = sim::Fixed::from_int(mapped.units_air_fighter);
        country.military.units_air_bomber = sim::Fixed::from_int(mapped.units_air_bomber);
        country.military.units_naval_surface = sim::Fixed::from_int(mapped.units_naval_surface);
        country.military.units_naval_submarine = sim::Fixed::from_int(mapped.units_naval_submarine);
        country.economic_stability = sim::Fixed::from_int(c.economic_stability);
        country.civilian_morale = sim::Fixed::from_int(c.civilian_morale);
        country.logistics_capacity = sim::Fixed::from_int(c.logistics_capacity);
        country.intelligence_level = sim::Fixed::from_int(c.intelligence_level);
        country.industrial_output = sim::Fixed::from_int(c.industrial_output);
        country.industrial_capital = sim::Fixed::from_double(std::clamp(c.industrial_output * 1.18, 8.0, 140.0));
        country.labor_participation = sim::Fixed::from_double(std::clamp(0.52 + c.civilian_morale / 420.0, 0.35, 0.90));
        country.technology_multiplier = sim::Fixed::from_double(std::clamp(0.55 + c.technology_level / 100.0, 0.40, 2.20));
        country.gdp_output = sim::Fixed::from_int(c.industrial_output);
        country.technology_level = sim::Fixed::from_int(c.technology_level);
        country.resource_reserve = sim::Fixed::from_int(c.resource_reserve);
        country.supply_level = sim::Fixed::from_int(c.supply_level);
        country.supply_capacity = sim::Fixed::from_int(c.supply_capacity);
        country.weather_severity = sim::Fixed::from_int(20 + static_cast<int64_t>(c.id % 5) * 10);
        country.seasonal_effect = sim::Fixed::from_int(45 + static_cast<int64_t>(c.id % 4) * 10);
        country.supply_stockpile = sim::Fixed::from_int(std::clamp<int64_t>(c.resource_reserve - 5, 20, 95));
        country.terrain.mountains = sim::Fixed::from_double(0.15 + 0.04 * static_cast<double>(c.id % 4));
        country.terrain.forests = sim::Fixed::from_double(0.22 + 0.05 * static_cast<double>((c.id + 1) % 4));
        country.terrain.urban = sim::Fixed::from_double(0.20 + 0.03 * static_cast<double>((c.id + 2) % 4));
        country.technology.missile_defense = sim::Fixed::from_int(std::clamp<int64_t>(c.technology_level + 4, 0, 100));
        country.technology.cyber_warfare = sim::Fixed::from_int(std::clamp<int64_t>(c.intelligence_level + 3, 0, 100));
        country.technology.electronic_warfare = sim::Fixed::from_int(std::clamp<int64_t>((c.intelligence_level + c.technology_level) / 2, 0, 100));
        country.technology.drone_operations = sim::Fixed::from_int(std::clamp<int64_t>((c.technology_level + c.logistics_capacity) / 2, 0, 100));
        country.resources.oil = sim::Fixed::from_int(std::clamp<int64_t>(c.resource_reserve + ((c.id % 3 == 0) ? 8 : -4), 0, 100));
        country.resources.minerals = sim::Fixed::from_int(std::clamp<int64_t>(c.resource_reserve + ((c.id % 3 == 1) ? 10 : -2), 0, 100));
        country.resources.food = sim::Fixed::from_int(std::clamp<int64_t>(c.resource_reserve + ((c.id % 3 == 2) ? 7 : 1), 0, 100));
        country.resources.rare_earth = sim::Fixed::from_int(std::clamp<int64_t>(c.resource_reserve - 8 + static_cast<int64_t>(c.id % 4) * 4, 0, 100));
        country.politics.government_stability = sim::Fixed::from_int(std::clamp<int64_t>(c.economic_stability - 4, 0, 100));
        country.politics.public_dissent = sim::Fixed::from_int(std::clamp<int64_t>(100 - c.civilian_morale + 8, 0, 100));
        country.politics.corruption = sim::Fixed::from_int(15 + static_cast<int64_t>(c.id % 5) * 7);
        country.nuclear_readiness = sim::Fixed::from_int(28 + static_cast<int64_t>(c.id % 5) * 12);
        country.deterrence_posture = sim::Fixed::from_int(35 + static_cast<int64_t>(c.id % 4) * 12);
        country.reputation = sim::Fixed::from_int(c.reputation);
        country.escalation_level = sim::Fixed::from_int(c.escalation_level);
        country.import_price_index = sim::Fixed::from_double(1.0);
        country.war_economy_intensity = sim::Fixed::from_double(std::clamp(12.0 + c.diplomatic_stance * 12.0, 0.0, 100.0));
        country.industrial_decay = sim::Fixed::from_int(0);
        country.debt_to_gdp = sim::Fixed::from_double(std::clamp(14.0 + static_cast<double>(c.id % 5) * 4.5, 0.0, 100.0));
        country.war_bond_stock = sim::Fixed::from_int(0);
        country.second_strike_capable = c.second_strike_capable;
        country.has_defense_pact_with = c.defense_pacts;
        country.has_non_aggression_with = c.non_aggression_pacts;
        country.has_trade_treaty_with = c.trade_treaties;
        for (uint16_t neighbor_id : c.adjacent) {
            country.trust_scores[neighbor_id] = sim::Fixed::from_int(45 + static_cast<int64_t>((c.id + neighbor_id) % 4) * 8);
        }
        for (const auto& kv : c.intel_on_enemy) {
            country.intel_on_enemy[kv.first] = sim::Fixed::from_int(kv.second);
        }
        country.diplomatic_stance = static_cast<sim::DiplomaticStance>(std::min<uint8_t>(2, c.diplomatic_stance));
        world.add_country(country);
    }

    return world;
}

battle::ModelManager model_manager_from_scenario(
    const ScenarioConfig& config,
    const std::map<std::string, std::string>& controller_overrides,
    uint32_t distributed_node_id,
    uint32_t distributed_total_nodes) {
    std::map<std::string, ScenarioModelProfile> profile_by_name;
    for (const ScenarioModelProfile& profile : config.models) {
        profile_by_name[profile.name] = profile;
    }

    battle::ModelManager manager;
    for (const ScenarioCountryConfig& country : config.countries) {
        std::string controller = country.controller;
        auto override_it = controller_overrides.find(controller);
        if (override_it != controller_overrides.end()) {
            controller = override_it->second;
        }

        if (controller.empty()) {
            throw std::runtime_error("country " + std::to_string(country.id) + " has no controller/model selected");
        }

        const auto profile_it = profile_by_name.find(controller);
        if (profile_it == profile_by_name.end()) {
            throw std::runtime_error("country " + std::to_string(country.id) + " references unknown model profile: " + controller);
        }

        battle::ManagedModel mm;
        mm.name = controller + "_c" + std::to_string(country.id);
        mm.team = country.team;
        mm.configured_model_path = profile_it->second.model_path;
        mm.controlled_country_ids = {country.id};

        if (mm.team.empty()) {
            mm.team = profile_it->second.team.empty() ? ("country_" + std::to_string(country.id)) : profile_it->second.team;
        }
        mm.model = create_model_instance(profile_it->second.model_path);

        manager.add_model(mm);
    }

    manager.set_distributed_partition(distributed_node_id, distributed_total_nodes);
    return manager;
}
