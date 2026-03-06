#ifndef TOURNAMENT_H
#define TOURNAMENT_H

#include "scenario_config.h"

#include <cstdint>
#include <string>
#include <vector>

struct TournamentMatchResult {
    std::string model_a;
    std::string model_b;
    std::string team_a;
    std::string team_b;
    uint64_t ticks = 0;
    double score_a = 0.0;
    double score_b = 0.0;
    std::string winner;
};

struct LeaderboardEntry {
    std::string model;
    uint32_t wins = 0;
    uint32_t losses = 0;
    uint32_t draws = 0;
    uint32_t matches = 0;
    uint32_t points = 0;
    double win_rate = 0.0;
};

struct TournamentResult {
    std::vector<TournamentMatchResult> matches;
    std::vector<LeaderboardEntry> leaderboard;
    std::string json;
};

TournamentResult run_round_robin_tournament(const ScenarioConfig& config, uint32_t rounds);
bool write_tournament_json(const std::string& path, const TournamentResult& result);

#endif
