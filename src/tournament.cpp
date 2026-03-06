#include "tournament.h"

#include "battle_runtime.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace {

double team_score(const sim::World& world, const battle::ModelManager& manager, const std::string& team) {
    double score = 0.0;
    for (const sim::Country& country : world.countries()) {
        if (manager.team_for_country(country.id) != team) {
            continue;
        }
        score += static_cast<double>(country.territory_cells) * 10.0;
        score += country.military.weighted_total().to_double();
        score += country.economic_stability.to_double() * 5.0;
        score += country.civilian_morale.to_double() * 5.0;
    }
    return score;
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

std::string result_json(const TournamentResult& result) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"matches\":[";
    for (size_t i = 0; i < result.matches.size(); ++i) {
        const auto& m = result.matches[i];
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"model_a\":\"" << json_escape(m.model_a) << "\",";
        oss << "\"model_b\":\"" << json_escape(m.model_b) << "\",";
        oss << "\"team_a\":\"" << json_escape(m.team_a) << "\",";
        oss << "\"team_b\":\"" << json_escape(m.team_b) << "\",";
        oss << "\"ticks\":" << m.ticks << ',';
        oss << "\"score_a\":" << std::llround(m.score_a) << ',';
        oss << "\"score_b\":" << std::llround(m.score_b) << ',';
        oss << "\"winner\":\"" << json_escape(m.winner) << "\"";
        oss << "}";
    }
    oss << "],";

    oss << "\"leaderboard\":[";
    for (size_t i = 0; i < result.leaderboard.size(); ++i) {
        const auto& row = result.leaderboard[i];
        if (i > 0) {
            oss << ',';
        }
        oss << "{";
        oss << "\"rank\":" << (i + 1) << ',';
        oss << "\"model\":\"" << json_escape(row.model) << "\",";
        oss << "\"matches\":" << row.matches << ',';
        oss << "\"wins\":" << row.wins << ',';
        oss << "\"losses\":" << row.losses << ',';
        oss << "\"draws\":" << row.draws << ',';
        oss << "\"points\":" << row.points << ',';
        oss << "\"win_rate\":" << row.win_rate;
        oss << "}";
    }
    oss << "]}";
    return oss.str();
}

}  // namespace

TournamentResult run_round_robin_tournament(const ScenarioConfig& config, uint32_t rounds) {
    TournamentResult result;

    std::set<std::string> team_set;
    for (const auto& country : config.countries) {
        if (!country.team.empty()) {
            team_set.insert(country.team);
        }
    }

    std::vector<std::string> teams(team_set.begin(), team_set.end());
    if (teams.size() < 2) {
        result.json = "{\"error\":\"scenario requires at least two teams\"}";
        return result;
    }

    const std::string team_a = teams[0];
    const std::string team_b = teams[1];

    std::vector<ScenarioModelProfile> models = config.models;
    if (models.size() < 2) {
        result.json = "{\"error\":\"tournament requires at least two configured model profiles\"}";
        return result;
    }

    for (const auto& model : models) {
        if (model.name.empty() || model.model_path.empty()) {
            result.json = "{\"error\":\"tournament requires every model profile to include a non-empty name and model_path\"}";
            return result;
        }
    }

    std::map<std::string, LeaderboardEntry> table;
    for (const auto& model : models) {
        LeaderboardEntry row;
        row.model = model.name;
        table[model.name] = row;
    }

    for (size_t i = 0; i < models.size(); ++i) {
        for (size_t j = i + 1; j < models.size(); ++j) {
            for (uint32_t round = 0; round < std::max<uint32_t>(1, rounds); ++round) {
                for (int side = 0; side < 2; ++side) {
                    const std::string model_for_team_a = side == 0 ? models[i].name : models[j].name;
                    const std::string model_for_team_b = side == 0 ? models[j].name : models[i].name;

                    std::map<std::string, std::string> overrides;
                    for (const auto& country : config.countries) {
                        if (country.team == team_a) {
                            overrides[country.controller] = model_for_team_a;
                        } else if (country.team == team_b) {
                            overrides[country.controller] = model_for_team_b;
                        }
                    }

                    sim::World world = world_from_scenario(config);
                    battle::ModelManager manager;
                    try {
                        manager = model_manager_from_scenario(config, overrides, 0, 1);
                    } catch (const std::exception& ex) {
                        result.json = std::string("{\"error\":\"") + json_escape(ex.what()) + "\"}";
                        return result;
                    }

                    std::vector<uint16_t> missing;
                    for (const auto& country : config.countries) {
                        if (!manager.has_loaded_model_for_country(country.id)) {
                            missing.push_back(country.id);
                        }
                    }
                    if (!missing.empty()) {
                        std::ostringstream oss;
                        oss << "tournament blocked: missing loaded model for country ids ";
                        for (size_t k = 0; k < missing.size(); ++k) {
                            if (k > 0) {
                                oss << ',';
                            }
                            oss << missing[k];
                        }
                        result.json = std::string("{\"error\":\"") + json_escape(oss.str()) + "\"}";
                        return result;
                    }

                    for (uint64_t t = 0; t < std::max<uint64_t>(1, config.ticks_per_match); ++t) {
                        auto decisions = manager.gather_decisions(world);
                        manager.coordinate_and_message(world, &decisions);
                        manager.apply_decisions(world, decisions);
                        world.run_tick();
                    }

                    const double score_a = team_score(world, manager, team_a);
                    const double score_b = team_score(world, manager, team_b);

                    TournamentMatchResult match;
                    match.model_a = model_for_team_a;
                    match.model_b = model_for_team_b;
                    match.team_a = team_a;
                    match.team_b = team_b;
                    match.ticks = config.ticks_per_match;
                    match.score_a = score_a;
                    match.score_b = score_b;

                    const double margin = std::abs(score_a - score_b);
                    const double tie_threshold = 0.02 * std::max(1.0, std::max(score_a, score_b));
                    if (margin <= tie_threshold) {
                        match.winner = "draw";
                        table[model_for_team_a].draws += 1;
                        table[model_for_team_b].draws += 1;
                        table[model_for_team_a].points += 1;
                        table[model_for_team_b].points += 1;
                    } else if (score_a > score_b) {
                        match.winner = model_for_team_a;
                        table[model_for_team_a].wins += 1;
                        table[model_for_team_a].points += 3;
                        table[model_for_team_b].losses += 1;
                    } else {
                        match.winner = model_for_team_b;
                        table[model_for_team_b].wins += 1;
                        table[model_for_team_b].points += 3;
                        table[model_for_team_a].losses += 1;
                    }

                    table[model_for_team_a].matches += 1;
                    table[model_for_team_b].matches += 1;
                    result.matches.push_back(std::move(match));
                }
            }
        }
    }

    for (auto& it : table) {
        LeaderboardEntry row = it.second;
        if (row.matches > 0) {
            row.win_rate = static_cast<double>(row.wins) / static_cast<double>(row.matches);
        }
        result.leaderboard.push_back(std::move(row));
    }

    std::sort(result.leaderboard.begin(), result.leaderboard.end(), [](const LeaderboardEntry& a, const LeaderboardEntry& b) {
        if (a.points == b.points) {
            if (a.wins == b.wins) {
                return a.model < b.model;
            }
            return a.wins > b.wins;
        }
        return a.points > b.points;
    });

    result.json = result_json(result);
    return result;
}

bool write_tournament_json(const std::string& path, const TournamentResult& result) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << result.json;
    return static_cast<bool>(out);
}
