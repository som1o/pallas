#include "battle_server.h"
#include "scenario_config.h"
#include "tournament.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (!part.empty()) {
            out.push_back(part);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    uint16_t port = 8080;
    std::string web_root = "../web";
    std::string replay_path = "../logs/battle_replay.bin";
    std::string scenario_path;
    bool tournament_mode = false;
    uint32_t tournament_rounds = 1;
    std::string tournament_output = "../logs/tournament_results.json";
    uint32_t distributed_node_id = 0;
    uint32_t distributed_total_nodes = 1;
    std::string distributed_bind_host = "0.0.0.0";
    uint16_t distributed_bind_port = 19090;
    uint32_t distributed_timeout_ms = 40;
    std::vector<std::string> distributed_peers;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--web-root" && i + 1 < argc) {
            web_root = argv[++i];
        } else if (arg == "--replay" && i + 1 < argc) {
            replay_path = argv[++i];
        } else if (arg == "--scenario" && i + 1 < argc) {
            scenario_path = argv[++i];
        } else if (arg == "--tournament") {
            tournament_mode = true;
        } else if (arg == "--tournament-rounds" && i + 1 < argc) {
            tournament_rounds = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--tournament-output" && i + 1 < argc) {
            tournament_output = argv[++i];
        } else if (arg == "--distributed-node-id" && i + 1 < argc) {
            distributed_node_id = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--distributed-total-nodes" && i + 1 < argc) {
            distributed_total_nodes = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--distributed-bind-host" && i + 1 < argc) {
            distributed_bind_host = argv[++i];
        } else if (arg == "--distributed-bind-port" && i + 1 < argc) {
            distributed_bind_port = static_cast<uint16_t>(std::stoul(argv[++i]));
        } else if (arg == "--distributed-timeout-ms" && i + 1 < argc) {
            distributed_timeout_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--distributed-peers" && i + 1 < argc) {
            distributed_peers = split_csv(argv[++i]);
        }
    }

    ScenarioConfig scenario = default_scenario_config();
    if (!scenario_path.empty()) {
        std::string error;
        if (!load_scenario_config(scenario_path, &scenario, &error)) {
            std::cerr << error << std::endl;
            return 1;
        }
    }

    if (tournament_mode) {
        TournamentResult tournament = run_round_robin_tournament(scenario, std::max<uint32_t>(1, tournament_rounds));
        if (!write_tournament_json(tournament_output, tournament)) {
            std::cerr << "failed to write tournament results: " << tournament_output << std::endl;
            return 1;
        }
        std::cout << "Tournament complete. Results: " << tournament_output << std::endl;
        for (size_t i = 0; i < tournament.leaderboard.size() && i < 10; ++i) {
            const auto& row = tournament.leaderboard[i];
            std::cout << (i + 1) << ". " << row.model
                      << " points=" << row.points
                      << " W-L-D=" << row.wins << "-" << row.losses << "-" << row.draws
                      << std::endl;
        }
        return 0;
    }

    sim::World world = world_from_scenario(scenario);
    battle::ModelManager manager;
    try {
        manager = model_manager_from_scenario(
            scenario,
            std::map<std::string, std::string>{},
            distributed_node_id,
            std::max<uint32_t>(1, distributed_total_nodes));
    } catch (const std::exception& ex) {
        std::cerr << "failed to configure model manager: " << ex.what() << std::endl;
        return 1;
    }

    battle::BattleEngine engine(std::move(world), std::move(manager));
    engine.set_mode(battle::SimulationMode::TurnBased);
    engine.set_tick_rate(4.0);
    engine.enable_replay_logging(replay_path);

    battle::DistributedRuntimeConfig dist_cfg;
    dist_cfg.node_id = distributed_node_id;
    dist_cfg.total_nodes = std::max<uint32_t>(1, distributed_total_nodes);
    dist_cfg.bind_host = distributed_bind_host;
    dist_cfg.bind_port = distributed_bind_port;
    dist_cfg.peer_endpoints = distributed_peers;
    dist_cfg.receive_timeout_ms = std::max<uint32_t>(5, distributed_timeout_ms);

    std::string distributed_error;
    if (!engine.configure_distributed_core(dist_cfg, &distributed_error)) {
        std::cerr << "failed to configure distributed core: " << distributed_error << std::endl;
        return 1;
    }

    battle::BattleServer server(engine, web_root, port);

    std::cout << "Battle server listening on http://127.0.0.1:" << port << std::endl;
    std::cout << "Use controls for turn-based step or continuous mode. Replay log: " << replay_path << std::endl;
    if (!scenario_path.empty()) {
        std::cout << "Scenario loaded: " << scenario_path << std::endl;
    }
    std::cout << "Distributed core active: node " << distributed_node_id
              << " / " << std::max<uint32_t>(1, distributed_total_nodes)
              << ", UDP " << distributed_bind_host << ':' << distributed_bind_port
              << std::endl;
    if (distributed_total_nodes > 1) {
        std::cout << "Distributed partition active: node " << distributed_node_id
                  << " / " << distributed_total_nodes << std::endl;
    }

    if (!server.run()) {
        std::cerr << "failed to run battle server" << std::endl;
        return 1;
    }
    return 0;
}
