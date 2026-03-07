#include "battle_common.h"
#include "scenario_config.h"
#include "data_pipeline.h"
#include "dataloader.h"
#include "logging.h"
#include "model.h"
#include "train_config.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

namespace {

std::atomic<bool> g_shutdown_requested{false};

void handle_signal(int) {
    g_shutdown_requested.store(true);
}

void install_signal_handlers() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
}

struct CliOptions {
    bool train_only = false;
    bool resume = false;
    bool rebuild_battle_data = false;
    std::string resume_path = "../data/best_state.bin";
    std::string model_zoo_dir = "../data/model_zoo";
    std::string log_dir = "../logs";
    std::string data_dir = "../data";
    std::string inspect_model_path;
    bool benchmark_only = false;
    std::string benchmark_bank_path;
    std::string benchmark_report_path;
    std::string benchmark_model_path;
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--train-only") {
            cli.train_only = true;
        } else if (arg == "--resume") {
            cli.resume = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cli.resume_path = argv[++i];
            }
        } else if (arg == "--rebuild-battle-data") {
            cli.rebuild_battle_data = true;
        } else if (arg == "--model-zoo-dir" && i + 1 < argc) {
            cli.model_zoo_dir = argv[++i];
        } else if (arg == "--log-dir" && i + 1 < argc) {
            cli.log_dir = argv[++i];
        } else if (arg == "--inspect-model" && i + 1 < argc) {
            cli.inspect_model_path = argv[++i];
        } else if (arg == "--data-dir" && i + 1 < argc) {
            cli.data_dir = argv[++i];
        } else if (arg == "--benchmark-only") {
            cli.benchmark_only = true;
        } else if (arg == "--benchmark-bank" && i + 1 < argc) {
            cli.benchmark_bank_path = argv[++i];
        } else if (arg == "--benchmark-report" && i + 1 < argc) {
            cli.benchmark_report_path = argv[++i];
        } else if (arg == "--benchmark-model" && i + 1 < argc) {
            cli.benchmark_model_path = argv[++i];
        }
    }
    if (const char* env_data = std::getenv("PALLAS_DATA_DIR")) {
        if (*env_data != '\0') {
            cli.data_dir = env_data;
        }
    }
    return cli;
}

struct BenchmarkBaseline {
    uint64_t initial_population = 1;
    float initial_economy = 0.0f;
    float initial_diplomacy = 0.5f;
};

struct BenchmarkAggregate {
    size_t scenarios = 0;
    size_t wins = 0;
    float reward_sum = 0.0f;
    float rank_sum = 0.0f;
};

float clamp_unit(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float clamp_signed(float value) {
    return std::clamp(value, -1.0f, 1.0f);
}

float norm_percent(int64_t milli) {
    return clamp_unit(static_cast<float>(milli) / 100000.0f);
}

float average_trust_norm(const sim::Country& country) {
    if (country.trust_scores.empty()) {
        return 0.5f;
    }
    double total = 0.0;
    for (const auto& kv : country.trust_scores) {
        total += norm_percent(kv.second.raw());
    }
    return clamp_unit(static_cast<float>(total / static_cast<double>(country.trust_scores.size())));
}

float diplomacy_score(const sim::Country& country) {
    return clamp_unit((norm_percent(country.reputation.raw()) + average_trust_norm(country)) * 0.5f);
}

float benchmark_reward(const sim::Country& country,
                       const BenchmarkBaseline& baseline,
                       uint32_t total_cells) {
    const float territory = clamp_signed(2.0f * (static_cast<float>(country.territory_cells) /
                                                static_cast<float>(std::max<uint32_t>(1, total_cells))) - 1.0f);
    const float economy = clamp_signed((norm_percent(country.economic_stability.raw()) - baseline.initial_economy) * 2.0f);
    const float population = clamp_signed(static_cast<float>(
        (static_cast<double>(std::max<uint64_t>(1, country.population)) /
         static_cast<double>(std::max<uint64_t>(1, baseline.initial_population))) - 1.0));
    const float diplomacy = clamp_signed((diplomacy_score(country) - baseline.initial_diplomacy) * 2.0f);
    return 0.35f * territory + 0.25f * economy + 0.25f * population + 0.15f * diplomacy;
}

battle::ModelManager build_benchmark_model_manager(const ScenarioConfig& scenario,
                                                    const Model& trained_model,
                                                    const ModelConfig& model_config) {
    battle::ModelManager manager;
    for (const ScenarioCountryConfig& country : scenario.countries) {
        auto replica = std::make_shared<Model>(battle_common::kBattleInputDim,
                                               battle_common::kBattleOutputDim,
                                               model_config);
        replica->copy_parameters_from(trained_model);
        replica->set_training(false);
        replica->set_inference_only(true);

        battle::ManagedModel managed;
        const std::string slot_name = country.controller.empty()
            ? ("benchmark_agent_" + std::to_string(country.id))
            : country.controller;
        managed.name = slot_name + "_c" + std::to_string(country.id);
        managed.team = country.team.empty() ? ("team_" + std::to_string(country.id)) : country.team;
        managed.model = std::move(replica);
        managed.controlled_country_ids = {country.id};
        manager.add_model(managed);
    }
    return manager;
}

bool run_scenario_bank_benchmark(const std::string& bank_path,
                                 const std::string& report_path,
                                 const Model& model,
                                 const ModelConfig& model_config,
                                 std::string* error_message) {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::exists(bank_path, ec) || !fs::is_directory(bank_path, ec)) {
        if (error_message != nullptr) {
            *error_message = "benchmark bank path is missing or not a directory: " + bank_path;
        }
        return false;
    }

    std::vector<std::string> scenario_files;
    for (const auto& entry : fs::directory_iterator(bank_path, ec)) {
        if (ec || !entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".json") {
            scenario_files.push_back(entry.path().string());
        }
    }
    std::sort(scenario_files.begin(), scenario_files.end());

    if (scenario_files.empty()) {
        if (error_message != nullptr) {
            *error_message = "no benchmark scenarios (*.json) found in: " + bank_path;
        }
        return false;
    }

    nlohmann::json report;
    report["generated_at_unix"] = static_cast<uint64_t>(std::time(nullptr));
    report["bank_path"] = bank_path;
    report["scenarios"] = nlohmann::json::array();

    std::unordered_map<std::string, BenchmarkAggregate> aggregate_by_model;
    uint64_t total_ticks_simulated = 0;

    for (const std::string& scenario_path : scenario_files) {
        ScenarioConfig scenario;
        std::string load_error;
        if (!load_scenario_config(scenario_path, &scenario, &load_error)) {
            if (error_message != nullptr) {
                *error_message = "failed to load scenario '" + scenario_path + "': " + load_error;
            }
            return false;
        }

        sim::World world = world_from_scenario(scenario);
        battle::ModelManager manager = build_benchmark_model_manager(scenario, model, model_config);

        std::unordered_map<uint16_t, BenchmarkBaseline> baselines;
        uint32_t total_cells = 0;
        for (const sim::Country& country : world.countries()) {
            BenchmarkBaseline baseline;
            baseline.initial_population = std::max<uint64_t>(1, country.population);
            baseline.initial_economy = norm_percent(country.economic_stability.raw());
            baseline.initial_diplomacy = diplomacy_score(country);
            baselines[country.id] = baseline;
            total_cells += country.territory_cells;
        }
        if (total_cells == 0) {
            total_cells = std::max<uint32_t>(1, scenario.map_width * scenario.map_height);
        }

        const uint64_t ticks = std::max<uint64_t>(1, scenario.ticks_per_match);
        for (uint64_t tick = 0; tick < ticks; ++tick) {
            auto decisions = manager.gather_decisions(world);
            manager.coordinate_and_message(world, &decisions);
            manager.apply_decisions(world, decisions);
            world.run_tick();
            ++total_ticks_simulated;
        }

        std::unordered_map<std::string, std::vector<float>> reward_by_model;
        for (const sim::Country& country : world.countries()) {
            auto it = baselines.find(country.id);
            if (it == baselines.end()) {
                continue;
            }
            const float reward = benchmark_reward(country, it->second, total_cells);
            const std::string model_name = manager.model_for_country(country.id);
            reward_by_model[model_name].push_back(reward);
        }

        struct RankedScore {
            std::string model_name;
            float reward = 0.0f;
        };
        std::vector<RankedScore> ranked;
        ranked.reserve(reward_by_model.size());
        for (const auto& kv : reward_by_model) {
            float avg = 0.0f;
            for (float reward : kv.second) {
                avg += reward;
            }
            avg /= static_cast<float>(std::max<size_t>(1, kv.second.size()));
            ranked.push_back({kv.first, avg});
        }

        std::sort(ranked.begin(), ranked.end(), [](const RankedScore& a, const RankedScore& b) {
            if (a.reward == b.reward) {
                return a.model_name < b.model_name;
            }
            return a.reward > b.reward;
        });

        if (ranked.empty()) {
            if (error_message != nullptr) {
                *error_message = "scenario produced no model scores: " + scenario_path;
            }
            return false;
        }

        nlohmann::json score_entries = nlohmann::json::array();
        for (size_t rank = 0; rank < ranked.size(); ++rank) {
            const RankedScore& entry = ranked[rank];
            BenchmarkAggregate& agg = aggregate_by_model[entry.model_name];
            agg.scenarios += 1;
            agg.reward_sum += entry.reward;
            agg.rank_sum += static_cast<float>(rank + 1);
            if (rank == 0) {
                agg.wins += 1;
            }

            nlohmann::json j;
            j["model"] = entry.model_name;
            j["reward"] = entry.reward;
            j["rank"] = rank + 1;
            score_entries.push_back(std::move(j));
        }

        nlohmann::json scenario_result;
        scenario_result["scenario_path"] = scenario_path;
        scenario_result["scenario_id"] = fs::path(scenario_path).stem().string();
        scenario_result["ticks"] = ticks;
        scenario_result["winner_model"] = ranked.front().model_name;
        scenario_result["winner_reward"] = ranked.front().reward;
        scenario_result["scores"] = std::move(score_entries);
        report["scenarios"].push_back(std::move(scenario_result));
    }

    std::vector<nlohmann::json> leaderboard_rows;
    leaderboard_rows.reserve(aggregate_by_model.size());
    for (const auto& kv : aggregate_by_model) {
        const BenchmarkAggregate& agg = kv.second;
        nlohmann::json row;
        row["model"] = kv.first;
        row["scenarios"] = agg.scenarios;
        row["wins"] = agg.wins;
        row["avg_reward"] = agg.scenarios > 0 ? (agg.reward_sum / static_cast<float>(agg.scenarios)) : 0.0f;
        row["avg_rank"] = agg.scenarios > 0 ? (agg.rank_sum / static_cast<float>(agg.scenarios)) : 0.0f;
        leaderboard_rows.push_back(std::move(row));
    }

    std::sort(leaderboard_rows.begin(), leaderboard_rows.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
        const uint64_t a_wins = a.value("wins", 0ULL);
        const uint64_t b_wins = b.value("wins", 0ULL);
        if (a_wins != b_wins) {
            return a_wins > b_wins;
        }
        const double a_reward = a.value("avg_reward", 0.0);
        const double b_reward = b.value("avg_reward", 0.0);
        if (a_reward != b_reward) {
            return a_reward > b_reward;
        }
        return a.value("model", std::string()) < b.value("model", std::string());
    });

    report["scenario_count"] = scenario_files.size();
    report["ticks_simulated"] = total_ticks_simulated;
    report["leaderboard"] = leaderboard_rows;

    const fs::path report_fs_path(report_path);
    if (!report_fs_path.parent_path().empty()) {
        fs::create_directories(report_fs_path.parent_path(), ec);
    }

    std::ofstream out(report_path);
    if (!out) {
        if (error_message != nullptr) {
            *error_message = "failed to open benchmark report path: " + report_path;
        }
        return false;
    }
    out << report.dump(2) << "\n";

    fs::path tsv_path = report_fs_path;
    if (tsv_path.extension() == ".json") {
        tsv_path.replace_extension(".tsv");
    } else {
        tsv_path += ".tsv";
    }

    std::ofstream tsv_out(tsv_path.string());
    if (!tsv_out) {
        if (error_message != nullptr) {
            *error_message = "failed to open benchmark TSV path: " + tsv_path.string();
        }
        return false;
    }

    tsv_out << "model\tscenarios\twins\tavg_reward\tavg_rank\n";
    tsv_out << std::fixed << std::setprecision(6);
    for (const auto& row : leaderboard_rows) {
        if (!row.is_object()) {
            continue;
        }
        tsv_out << row.value("model", std::string()) << '\t'
                << row.value("scenarios", 0ULL) << '\t'
                << row.value("wins", 0ULL) << '\t'
                << row.value("avg_reward", 0.0) << '\t'
                << row.value("avg_rank", 0.0) << '\n';
    }

    return true;
}

bool ensure_directory(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(dir, ec)) {
        return fs::is_directory(dir, ec);
    }
    return fs::create_directories(dir, ec);
}

std::vector<float> load_battle_class_weights(const TrainConfig& train_config) {
    std::vector<float> weights(battle_common::kBattlePolicyActionDim, 1.0f);
    if (!train_config.use_class_weights) {
        return weights;
    }

    std::ifstream in(train_config.class_weights_path);
    if (!in) {
        return weights;
    }

    std::unordered_map<std::string, size_t> action_to_id = {
        {"attack", 0},
        {"defend", 1},
        {"negotiate", 2},
        {"surrender", 3},
        {"transfer_weapons", 4},
        {"focus_economy", 5},
        {"develop_technology", 6},
        {"form_alliance", 7},
        {"betray", 8},
        {"cyber_operation", 9},
        {"sign_trade_agreement", 10},
        {"cancel_trade_agreement", 11},
        {"impose_embargo", 12},
        {"invest_in_resource_extraction", 13},
        {"reduce_military_upkeep", 14},
        {"suppress_dissent", 15},
        {"hold_elections", 16},
        {"coup_attempt", 17},
        {"propose_defense_pact", 18},
        {"propose_non_aggression", 19},
        {"break_treaty", 20},
        {"request_intel", 21},
        {"deploy_units", 22},
        {"tactical_nuke", 23},
        {"strategic_nuke", 24},
        {"cyber_attack", 25},
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::string action;
        float weight = 1.0f;
        if (!(iss >> action >> weight)) {
            continue;
        }
        auto it = action_to_id.find(action);
        if (it != action_to_id.end()) {
            weights[it->second] = std::max(0.0f, weight);
        }
    }

    return weights;
}

struct Metrics {
    float loss = 0.0f;
    float top1 = 0.0f;
    float top3 = 0.0f;
    float top5 = 0.0f;
    size_t count = 0;
};

Tensor slice_logits(const Tensor& logits, size_t offset, size_t size) {
    const size_t capped = offset < logits.data.size() ? std::min(size, logits.data.size() - offset) : 0;
    Tensor out({1, capped}, 0.0f);
    for (size_t i = 0; i < capped; ++i) {
        out.data[i] = logits.data[offset + i];
    }
    return out;
}

uint32_t strategic_goal_from_action(uint32_t action) {
    switch (action) {
        case battle_common::kActionAttack:
        case battle_common::kActionDeployUnits:
        case battle_common::kActionCyberAttack:
        case battle_common::kActionCyberOperation:
        case battle_common::kActionTacticalNuke:
        case battle_common::kActionStrategicNuke:
        case battle_common::kActionBetray:
        case battle_common::kActionBreakTreaty:
            return 0;
        case battle_common::kActionDefend:
        case battle_common::kActionRequestIntel:
        case battle_common::kActionProposeDefensePact:
        case battle_common::kActionProposeNonAggression:
        case battle_common::kActionSuppressDissent:
        case battle_common::kActionTransferWeapons:
            return 1;
        case battle_common::kActionSignTradeAgreement:
        case battle_common::kActionCancelTradeAgreement:
        case battle_common::kActionImposeEmbargo:
        case battle_common::kActionInvestInResourceExtraction:
        case battle_common::kActionReduceMilitaryUpkeep:
            return 2;
        default:
            return 3;
    }
}

uint32_t target_bucket_from_features(const Tensor& sample) {
    const float threat = sample.data[55];
    const float treaty = sample.data[63];
    if (threat > 0.62f) {
        return 0;
    }
    if (treaty > 0.52f) {
        return 2;
    }
    return 1;
}

uint32_t commitment_bucket_from_features(const Tensor& sample) {
    const float attack_opportunity = sample.data[65];
    const float crisis_pressure = sample.data[66];
    const float score = attack_opportunity - 0.6f * crisis_pressure;
    if (score < 0.18f) {
        return 0;
    }
    if (score < 0.42f) {
        return 1;
    }
    return 2;
}

uint32_t allocation_bucket_from_features(const Tensor& sample) {
    const float attack_opportunity = sample.data[65];
    const float crisis_pressure = sample.data[66];
    const float reserve_stress = sample.data[78];
    if (attack_opportunity >= crisis_pressure && attack_opportunity >= reserve_stress) {
        return 0;
    }
    if (crisis_pressure >= reserve_stress) {
        return 1;
    }
    return 2;
}

uint32_t opponent_action_target_from_features(const Tensor& sample) {
    const float threat = sample.data[55];
    const float trust = sample.data[69];
    const float treaty = sample.data[63];
    if (threat > 0.62f && trust < 0.45f) {
        return battle_common::kActionAttack;
    }
    if (treaty > 0.58f && trust > 0.48f) {
        return battle_common::kActionProposeDefensePact;
    }
    return battle_common::kActionDefend;
}

float value_target_from_action(uint32_t action) {
    const uint32_t goal = strategic_goal_from_action(action);
    if (goal == 0) {
        return 0.75f;
    }
    if (goal == 1) {
        return 0.55f;
    }
    if (goal == 2) {
        return 0.50f;
    }
    return 0.60f;
}

float append_multitask_loss_and_grad(const Tensor& logits,
                                     const Tensor& sample,
                                     uint32_t action_target,
                                     float outcome_target,
                                     float reward_total,
                                     const TrainConfig& train_config,
                                     float action_weight,
                                     Tensor& grad_out) {
    grad_out = Tensor(logits.shape, 0.0f);

    const Tensor policy_logits = slice_logits(logits,
                                              battle_common::kBattleHeadPolicyOffset,
                                              battle_common::kBattlePolicyActionDim);
    float total_loss = 0.0f;

    if (train_config.use_actor_critic) {
        const float value_pred = std::tanh(logits.data[battle_common::kBattleHeadValueOffset]);
        const float scaled_reward = reward_total * train_config.reward_scale;
        const float advantage = std::clamp(scaled_reward - value_pred, -3.0f, 3.0f);

        std::vector<float> probs(policy_logits.data.size(), 0.0f);
        float max_logit = policy_logits.data.empty() ? 0.0f : policy_logits.data[0];
        for (size_t i = 1; i < policy_logits.data.size(); ++i) {
            max_logit = std::max(max_logit, policy_logits.data[i]);
        }
        float denom = 0.0f;
        for (size_t i = 0; i < policy_logits.data.size(); ++i) {
            probs[i] = std::exp(policy_logits.data[i] - max_logit);
            denom += probs[i];
        }
        denom = std::max(1e-7f, denom);
        for (float& p : probs) {
            p /= denom;
        }

        if (!probs.empty()) {
            const size_t safe_action = std::min<size_t>(action_target, probs.size() - 1);
            const float selected_prob = std::max(1e-7f, probs[safe_action]);
            total_loss += -train_config.policy_loss_weight * advantage * std::log(selected_prob) * action_weight;

            for (size_t i = 0; i < probs.size(); ++i) {
                const float one_hot = i == safe_action ? 1.0f : 0.0f;
                grad_out.data[battle_common::kBattleHeadPolicyOffset + i] =
                    train_config.policy_loss_weight * advantage * (probs[i] - one_hot) * action_weight;
            }
        }

        float entropy = 0.0f;
        for (float p : probs) {
            entropy += -p * std::log(std::max(1e-7f, p));
        }
        total_loss += -train_config.entropy_coeff * entropy;
    } else {
        const Tensor grad_policy = grad_cross_entropy_advanced(policy_logits,
                                                               action_target,
                                                               train_config.label_smoothing,
                                                               action_weight);
        for (size_t i = 0; i < grad_policy.data.size(); ++i) {
            grad_out.data[battle_common::kBattleHeadPolicyOffset + i] = grad_policy.data[i];
        }

        total_loss += cross_entropy_advanced(policy_logits,
                                             action_target,
                                             train_config.label_smoothing,
                                             action_weight);
    }

    const uint32_t goal_target = strategic_goal_from_action(action_target);
    const Tensor goal_logits = slice_logits(logits,
                                            battle_common::kBattleHeadStrategicOffset,
                                            battle_common::kBattleStrategicGoalDim);
    const Tensor grad_goal = grad_cross_entropy_advanced(goal_logits, goal_target, 0.0f, 0.20f);
    for (size_t i = 0; i < grad_goal.data.size(); ++i) {
        grad_out.data[battle_common::kBattleHeadStrategicOffset + i] = grad_goal.data[i];
    }
    total_loss += cross_entropy_advanced(goal_logits, goal_target, 0.0f, 0.20f);

    const uint32_t target_bucket = target_bucket_from_features(sample);
    const Tensor target_logits = slice_logits(logits,
                                              battle_common::kBattleHeadTargetBucketOffset,
                                              battle_common::kBattleTacticalTargetBucketDim);
    const Tensor grad_target = grad_cross_entropy_advanced(target_logits, target_bucket, 0.0f, 0.12f);
    for (size_t i = 0; i < grad_target.data.size(); ++i) {
        grad_out.data[battle_common::kBattleHeadTargetBucketOffset + i] = grad_target.data[i];
    }
    total_loss += cross_entropy_advanced(target_logits, target_bucket, 0.0f, 0.12f);

    const uint32_t commitment_bucket = commitment_bucket_from_features(sample);
    const Tensor commitment_logits = slice_logits(logits,
                                                  battle_common::kBattleHeadCommitmentOffset,
                                                  battle_common::kBattleCommitmentBucketDim);
    const Tensor grad_commit = grad_cross_entropy_advanced(commitment_logits, commitment_bucket, 0.0f, 0.12f);
    for (size_t i = 0; i < grad_commit.data.size(); ++i) {
        grad_out.data[battle_common::kBattleHeadCommitmentOffset + i] = grad_commit.data[i];
    }
    total_loss += cross_entropy_advanced(commitment_logits, commitment_bucket, 0.0f, 0.12f);

    const uint32_t alloc_bucket = allocation_bucket_from_features(sample);
    const Tensor alloc_logits = slice_logits(logits,
                                             battle_common::kBattleHeadAllocationOffset,
                                             battle_common::kBattleAllocationBucketDim);
    const Tensor grad_alloc = grad_cross_entropy_advanced(alloc_logits, alloc_bucket, 0.0f, 0.12f);
    for (size_t i = 0; i < grad_alloc.data.size(); ++i) {
        grad_out.data[battle_common::kBattleHeadAllocationOffset + i] = grad_alloc.data[i];
    }
    total_loss += cross_entropy_advanced(alloc_logits, alloc_bucket, 0.0f, 0.12f);

    const uint32_t opponent_target = opponent_action_target_from_features(sample);
    const Tensor opponent_logits = slice_logits(logits,
                                                battle_common::kBattleHeadOpponentOffset,
                                                battle_common::kBattleOpponentActionDim);
    const Tensor grad_opp = grad_cross_entropy_advanced(opponent_logits, opponent_target, 0.02f, 0.18f);
    for (size_t i = 0; i < grad_opp.data.size(); ++i) {
        grad_out.data[battle_common::kBattleHeadOpponentOffset + i] = grad_opp.data[i];
    }
    total_loss += cross_entropy_advanced(opponent_logits, opponent_target, 0.02f, 0.18f);

    if (battle_common::kBattleHeadValueOffset < logits.data.size()) {
        const float value_target = train_config.use_actor_critic
            ? std::clamp(outcome_target * train_config.reward_scale, -1.0f, 1.0f)
            : value_target_from_action(action_target);
        const float pred = std::tanh(logits.data[battle_common::kBattleHeadValueOffset]);
        const float diff = pred - value_target;
        total_loss += train_config.value_loss_weight * diff * diff;
        const float dtanh = 1.0f - pred * pred;
        grad_out.data[battle_common::kBattleHeadValueOffset] = 2.0f * train_config.value_loss_weight * diff * dtanh;
    }

    return total_loss;
}

Metrics evaluate_validation(Model& model,
                            BattleBatchLoader& loader,
                            const TrainConfig& train_config,
                            const std::vector<float>& class_weights) {
    model.set_training(false);
    loader.reset();

    Metrics m;
    Tensor inputs({1, battle_common::kBattleInputDim}, 0.0f);
    std::vector<uint32_t> targets;
    std::vector<float> outcomes;
    std::vector<float> rewards;
    while (loader.next(inputs, targets, outcomes, rewards)) {
        for (size_t i = 0; i < targets.size(); ++i) {
            Tensor sample({1, battle_common::kBattleInputDim}, 0.0f);
            const float* src = &inputs.data[i * battle_common::kBattleInputDim];
            std::copy(src, src + battle_common::kBattleInputDim, sample.data.begin());

            Tensor logits = model.forward(sample);
            Tensor policy_logits = slice_logits(logits,
                                                battle_common::kBattleHeadPolicyOffset,
                                                battle_common::kBattlePolicyActionDim);
            const uint32_t target = targets[i];
            const float weight = target < class_weights.size() ? class_weights[target] : 1.0f;
            Tensor throwaway_grad({1, battle_common::kBattleOutputDim}, 0.0f);
            const float outcome = i < outcomes.size() ? outcomes[i] : 0.0f;
            const float reward = i < rewards.size() ? rewards[i] : outcome;
            m.loss += append_multitask_loss_and_grad(logits, sample, target, outcome, reward, train_config, weight, throwaway_grad);
            m.top1 += top_k_hit(policy_logits, target, 1) ? 1.0f : 0.0f;
            m.top3 += top_k_hit(policy_logits, target, 3) ? 1.0f : 0.0f;
            m.top5 += top_k_hit(policy_logits, target, 5) ? 1.0f : 0.0f;
            m.count += 1;
        }
    }

    if (m.count > 0) {
        m.loss /= static_cast<float>(m.count);
        m.top1 /= static_cast<float>(m.count);
        m.top3 /= static_cast<float>(m.count);
        m.top5 /= static_cast<float>(m.count);
    }
    return m;
}

void render_progress(size_t epoch, size_t total_epochs, size_t step, size_t total_steps) {
    static const char spinner[] = {'|', '/', '-', '\\'};
    const char spin = spinner[step % 4];
    const double ratio = total_steps > 0 ? static_cast<double>(step) / static_cast<double>(total_steps) : 0.0;
    const int width = 34;
    const int filled = static_cast<int>(ratio * width);

    std::string bar;
    bar.reserve(static_cast<size_t>(width));
    for (int i = 0; i < width; ++i) {
        bar.push_back(i < filled ? '=' : ' ');
    }

    std::fprintf(stderr,
                 "\r%c Epoch %zu/%zu [%s] %6.2f%%",
                 spin,
                 epoch + 1,
                 total_epochs,
                 bar.c_str(),
                 ratio * 100.0);
    std::fflush(stderr);
}

void append_model_zoo_manifest(const std::string& zoo_dir, const ModelTrainingMetadata& meta, const std::string& path) {
    if (!ensure_directory(zoo_dir)) {
        return;
    }
    std::ofstream out(zoo_dir + "/manifest.tsv", std::ios::app);
    if (!out) {
        return;
    }
    out << meta.timestamp_unix << '\t'
        << meta.epoch << '\t'
        << meta.val_loss << '\t'
        << meta.val_top1 << '\t'
        << path << '\n';
}

void print_model_inspection(const std::string& path) {
    size_t in_dim = 0;
    size_t out_dim = 0;
    ModelConfig cfg;
    std::string err;

    if (!inspect_model_state(path, &in_dim, &out_dim, &cfg, &err)) {
        std::cerr << "inspect failed: " << err << "\n";
        return;
    }

    std::cout << "path: " << path << "\n";
    std::cout << "input_dim: " << in_dim << "\n";
    std::cout << "output_dim: " << out_dim << "\n";
    std::cout << "battle_compatible: "
              << ((in_dim == battle_common::kBattleInputDim && out_dim == battle_common::kBattleOutputDim) ? "yes" : "no")
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    install_signal_handlers();
    const CliOptions cli = parse_cli(argc, argv);

    if (!cli.inspect_model_path.empty()) {
        print_model_inspection(cli.inspect_model_path);
        return 0;
    }

    if (!ensure_directory(cli.log_dir)) {
        std::fprintf(stderr, "failed to create log directory\n");
        return 1;
    }

    if (!logging::init_logging(cli.log_dir, "pallas")) {
        std::fprintf(stderr, "failed to initialize logging\n");
        return 1;
    }

    const std::string config_path = cli.data_dir + "/model_config.json";
    const std::string train_config_path = cli.data_dir + "/train_config.json";

    {
        std::ifstream model_in(config_path);
        if (!model_in) {
            std::fprintf(stderr, "missing required config file: %s\n", config_path.c_str());
            return 1;
        }
    }
    {
        std::ifstream train_in(train_config_path);
        if (!train_in) {
            std::fprintf(stderr, "missing required config file: %s\n", train_config_path.c_str());
            return 1;
        }
    }

    ModelConfig model_config = load_model_config(config_path);
    TrainConfig train_config = load_train_config(train_config_path);

    std::cout << "using model config: " << config_path << "\n";
    std::cout << "using train config: " << train_config_path << "\n";

    std::string err;
    if (!validate_model_config(model_config, err)) {
        logging::log_event(logging::Level::Error, "model_config_invalid", {{"error", err}});
        return 1;
    }
    if (!validate_train_config(train_config, err)) {
        logging::log_event(logging::Level::Error, "train_config_invalid", {{"error", err}});
        return 1;
    }

    const std::string benchmark_bank_path = cli.benchmark_bank_path.empty()
        ? (cli.data_dir + "/scenario_bank")
        : cli.benchmark_bank_path;
    const std::string benchmark_report_path = cli.benchmark_report_path.empty()
        ? (cli.log_dir + "/scenario_benchmark.json")
        : cli.benchmark_report_path;

    if (cli.benchmark_only) {
        Model benchmark_model(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, model_config);
        benchmark_model.set_training(false);
        benchmark_model.set_inference_only(true);

        ModelFileInfo file_info;
        const std::string benchmark_model_path = cli.benchmark_model_path.empty()
            ? cli.resume_path
            : cli.benchmark_model_path;
        if (!benchmark_model.load_state(benchmark_model_path,
                                        battle_common::kBattleInputDim,
                                        battle_common::kBattleOutputDim,
                                        &file_info)) {
            logging::log_event(logging::Level::Error, "benchmark_model_load_failed", {
                {"path", benchmark_model_path}
            });
            return 1;
        }

        std::string benchmark_error;
        if (!run_scenario_bank_benchmark(benchmark_bank_path,
                                         benchmark_report_path,
                                         benchmark_model,
                                         model_config,
                                         &benchmark_error)) {
            logging::log_event(logging::Level::Error, "benchmark_bank_failed", {
                {"error", benchmark_error},
                {"bank_path", benchmark_bank_path},
                {"report_path", benchmark_report_path}
            });
            return 1;
        }

        logging::log_event(logging::Level::Info, "benchmark_bank_complete", {
            {"bank_path", benchmark_bank_path},
            {"report_path", benchmark_report_path}
        });
        std::cout << "benchmark report written: " << benchmark_report_path << "\n";
        logging::flush();
        return 0;
    }

    std::vector<BattleSample> samples;
    BattleDatasetConfig dataset_cfg = make_battle_dataset_config(cli.data_dir);
    try {
        const BattleDatasetInfo info = prepare_battle_dataset(dataset_cfg, samples, cli.rebuild_battle_data);
        logging::log_event(logging::Level::Info, "battle_dataset_ready", {
            {"sample_count", std::to_string(info.sample_count)},
            {"rebuilt", info.rebuilt ? "true" : "false"},
            {"json_path", info.json_path},
            {"txt_path", info.txt_path},
            {"vocab_path", info.vocab_path},
            {"input_dim", std::to_string(battle_common::kBattleInputDim)},
            {"output_dim", std::to_string(battle_common::kBattleOutputDim)}
        });
        std::cout << "battle dataset: " << (info.rebuilt ? "rebuilt" : "reused")
                  << " (" << info.sample_count << " rows)\n";
        std::cout << "json: " << info.json_path << "\n";
        std::cout << "txt: " << info.txt_path << "\n";
        std::cout << "vocab: " << info.vocab_path << "\n";
    } catch (const std::exception& ex) {
        logging::log_event(logging::Level::Error, "battle_data_prepare_failed", {{"error", ex.what()}});
        return 1;
    }

    if (samples.size() < 8) {
        logging::log_event(logging::Level::Error, "battle_dataset_too_small", {{"sample_count", std::to_string(samples.size())}});
        return 1;
    }

    std::vector<std::array<float, battle_common::kBattleInputDim>> features;
    std::vector<uint32_t> actions;
    std::vector<float> outcomes;
    std::vector<float> rewards;
    features.reserve(samples.size());
    actions.reserve(samples.size());
    outcomes.reserve(samples.size());
    rewards.reserve(samples.size());
    for (const BattleSample& s : samples) {
        features.push_back(s.features);
        actions.push_back(s.action);
        outcomes.push_back(s.outcome);
        rewards.push_back(s.reward_total);
    }

    const size_t total = samples.size();
    size_t val_count = static_cast<size_t>(static_cast<float>(total) * train_config.validation_split);
    val_count = std::max<size_t>(1, val_count);
    if (val_count >= total) {
        val_count = total - 1;
    }
    const size_t train_count = total - val_count;

    std::vector<size_t> all_indices(total);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::mt19937 split_rng(2026);
    std::shuffle(all_indices.begin(), all_indices.end(), split_rng);

    std::vector<size_t> train_indices(all_indices.begin(), all_indices.begin() + train_count);
    std::vector<size_t> val_indices(all_indices.begin() + train_count, all_indices.end());

    OptimizerConfig opt;
    opt.type = train_config.optimizer;
    opt.beta1 = train_config.adam_beta1;
    opt.beta2 = train_config.adam_beta2;
    opt.epsilon = train_config.adam_epsilon;
    opt.weight_decay = train_config.weight_decay;

    Model model(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, model_config);
    model.configure_optimizer(opt);
    model.set_training(true);

    if (cli.resume) {
        ModelFileInfo file_info;
        if (!model.load_state(cli.resume_path,
                              battle_common::kBattleInputDim,
                              battle_common::kBattleOutputDim,
                              &file_info)) {
            logging::log_event(logging::Level::Error, "resume_failed", {{"path", cli.resume_path}});
            return 1;
        }
    }

    BattleBatchLoader train_loader(&features, &actions, &outcomes, &rewards, train_indices, train_config.batch_size, true, 1234);
    BattleBatchLoader val_loader(&features, &actions, &outcomes, &rewards, val_indices, train_config.batch_size, false, 5678);
    std::vector<float> class_weights = load_battle_class_weights(train_config);

    const int omp_threads = std::max(1, omp_get_max_threads());
    const size_t grad_size = model.gradient_size();
    std::vector<Model> worker_models;
    worker_models.reserve(static_cast<size_t>(omp_threads));
    for (int t = 0; t < omp_threads; ++t) {
        worker_models.emplace_back(battle_common::kBattleInputDim, battle_common::kBattleOutputDim, model_config);
        worker_models.back().configure_optimizer(opt);
        worker_models.back().set_training(true);
    }

    std::vector<std::vector<float>> thread_grad_sums(
        static_cast<size_t>(omp_threads),
        std::vector<float>(grad_size, 0.0f));
    std::vector<float> reduced_grad(grad_size, 0.0f);

    float best_val_loss = std::numeric_limits<float>::infinity();
    size_t epochs_without_improve = 0;

    for (size_t epoch = 0; epoch < train_config.epochs; ++epoch) {
        if (g_shutdown_requested.load()) {
            break;
        }

        model.set_training(true);
        train_loader.reset();
        const float lr = scheduler_lr(train_config, epoch);

        Metrics train_metrics;
        const size_t total_steps = train_loader.steps_per_epoch();
        size_t step = 0;

        Tensor batch_inputs({1, battle_common::kBattleInputDim}, 0.0f);
        std::vector<uint32_t> targets;
        std::vector<float> batch_outcomes;
        std::vector<float> batch_rewards;
        while (train_loader.next(batch_inputs, targets, batch_outcomes, batch_rewards)) {
            if (g_shutdown_requested.load()) {
                break;
            }

            const size_t batch = targets.size();
            if (batch == 0) {
                continue;
            }

            for (int t = 0; t < omp_threads; ++t) {
                worker_models[static_cast<size_t>(t)].copy_parameters_from(model);
                std::fill(thread_grad_sums[static_cast<size_t>(t)].begin(),
                          thread_grad_sums[static_cast<size_t>(t)].end(),
                          0.0f);
            }

            std::vector<float> thread_loss(static_cast<size_t>(omp_threads), 0.0f);
            std::vector<float> thread_top1(static_cast<size_t>(omp_threads), 0.0f);
            std::vector<float> thread_top3(static_cast<size_t>(omp_threads), 0.0f);
            std::vector<float> thread_top5(static_cast<size_t>(omp_threads), 0.0f);
            std::vector<size_t> thread_count(static_cast<size_t>(omp_threads), 0);

#pragma omp parallel default(none) shared(batch, batch_inputs, targets, batch_outcomes, batch_rewards, class_weights, train_config, worker_models, thread_grad_sums, thread_loss, thread_top1, thread_top3, thread_top5, thread_count)
            {
                const int tid = omp_get_thread_num();
                Model& local_model = worker_models[static_cast<size_t>(tid)];
                std::vector<float>& grad_accum = thread_grad_sums[static_cast<size_t>(tid)];
                std::vector<float> grad_vec;

#pragma omp for schedule(static)
                for (size_t i = 0; i < batch; ++i) {
                    Tensor sample({1, battle_common::kBattleInputDim}, 0.0f);
                    const float* src = &batch_inputs.data[i * battle_common::kBattleInputDim];
                    std::copy(src, src + battle_common::kBattleInputDim, sample.data.begin());

                    local_model.zero_grad();
                    Tensor logits = local_model.forward(sample);
                    Tensor policy_logits = slice_logits(logits,
                                                        battle_common::kBattleHeadPolicyOffset,
                                                        battle_common::kBattlePolicyActionDim);
                    const uint32_t target = targets[i];
                    const float weight = target < class_weights.size() ? class_weights[target] : 1.0f;
                    const float outcome = i < batch_outcomes.size() ? batch_outcomes[i] : 0.0f;
                    const float reward = i < batch_rewards.size() ? batch_rewards[i] : outcome;

                    Tensor grad(logits.shape, 0.0f);
                    thread_loss[static_cast<size_t>(tid)] +=
                        append_multitask_loss_and_grad(logits, sample, target, outcome, reward, train_config, weight, grad);
                    thread_top1[static_cast<size_t>(tid)] += top_k_hit(policy_logits, target, 1) ? 1.0f : 0.0f;
                    thread_top3[static_cast<size_t>(tid)] += top_k_hit(policy_logits, target, 3) ? 1.0f : 0.0f;
                    thread_top5[static_cast<size_t>(tid)] += top_k_hit(policy_logits, target, 5) ? 1.0f : 0.0f;
                    thread_count[static_cast<size_t>(tid)] += 1;

                    local_model.backward(grad);
                    local_model.gradients_to_vector(grad_vec);
                    for (size_t g = 0; g < grad_accum.size(); ++g) {
                        grad_accum[g] += grad_vec[g];
                    }
                }
            }

            std::fill(reduced_grad.begin(), reduced_grad.end(), 0.0f);
            for (int t = 0; t < omp_threads; ++t) {
                const size_t ti = static_cast<size_t>(t);
                train_metrics.loss += thread_loss[ti];
                train_metrics.top1 += thread_top1[ti];
                train_metrics.top3 += thread_top3[ti];
                train_metrics.top5 += thread_top5[ti];
                train_metrics.count += thread_count[ti];
                for (size_t g = 0; g < reduced_grad.size(); ++g) {
                    reduced_grad[g] += thread_grad_sums[ti][g];
                }
            }

            const float inv_batch = 1.0f / static_cast<float>(batch);
            for (float& g : reduced_grad) {
                g *= inv_batch;
            }

            model.zero_grad();
            model.set_gradients_from_vector(reduced_grad);
            model.update(lr);

            ++step;
            render_progress(epoch, train_config.epochs, step, total_steps);
        }
        std::fprintf(stderr, "\n");

        if (train_metrics.count > 0) {
            train_metrics.loss /= static_cast<float>(train_metrics.count);
            train_metrics.top1 /= static_cast<float>(train_metrics.count);
            train_metrics.top3 /= static_cast<float>(train_metrics.count);
            train_metrics.top5 /= static_cast<float>(train_metrics.count);
        }

        Metrics val_metrics = evaluate_validation(model, val_loader, train_config, class_weights);

        logging::log_event(logging::Level::Info, "battle_epoch_complete", {
            {"epoch", std::to_string(epoch)},
            {"lr", std::to_string(lr)},
            {"train_loss", std::to_string(train_metrics.loss)},
            {"val_loss", std::to_string(val_metrics.loss)},
            {"train_top1", std::to_string(train_metrics.top1)},
            {"val_top1", std::to_string(val_metrics.top1)}
        });

        if (val_metrics.loss < best_val_loss) {
            best_val_loss = val_metrics.loss;
            epochs_without_improve = 0;

            ModelTrainingMetadata meta;
            meta.timestamp_unix = static_cast<uint64_t>(std::time(nullptr));
            meta.epoch = epoch;
            meta.val_loss = val_metrics.loss;
            meta.val_top1 = val_metrics.top1;
            meta.optimizer = opt.type;

            model.save_state(cli.resume_path,
                             battle_common::kBattleInputDim,
                             battle_common::kBattleOutputDim,
                             model_config,
                             meta);

            ensure_directory(cli.model_zoo_dir);
            const std::string zoo_path = cli.model_zoo_dir + "/battle_model_"
                + std::to_string(meta.timestamp_unix)
                + "_e" + std::to_string(epoch)
                + "_top1_" + std::to_string(val_metrics.top1)
                + ".bin";
            model.save_state(zoo_path,
                             battle_common::kBattleInputDim,
                             battle_common::kBattleOutputDim,
                             model_config,
                             meta);
            append_model_zoo_manifest(cli.model_zoo_dir, meta, zoo_path);
        } else {
            ++epochs_without_improve;
            if (epochs_without_improve >= train_config.early_stopping_patience) {
                logging::log_event(logging::Level::Info, "battle_early_stopping", {{"epoch", std::to_string(epoch)}});
                break;
            }
        }
    }

    logging::flush();

    if (!cli.benchmark_bank_path.empty()) {
        std::string benchmark_error;
        if (!run_scenario_bank_benchmark(benchmark_bank_path,
                                         benchmark_report_path,
                                         model,
                                         model_config,
                                         &benchmark_error)) {
            logging::log_event(logging::Level::Error, "benchmark_bank_failed", {
                {"error", benchmark_error},
                {"bank_path", benchmark_bank_path},
                {"report_path", benchmark_report_path}
            });
            return 1;
        }

        logging::log_event(logging::Level::Info, "benchmark_bank_complete", {
            {"bank_path", benchmark_bank_path},
            {"report_path", benchmark_report_path}
        });
        std::cout << "benchmark report written: " << benchmark_report_path << "\n";
        logging::flush();
    }

    return 0;
}
