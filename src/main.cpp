#include "battle_common.h"
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
#include <iostream>
#include <limits>
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
        }
    }
    if (const char* env_data = std::getenv("PALLAS_DATA_DIR")) {
        if (*env_data != '\0') {
            cli.data_dir = env_data;
        }
    }
    return cli;
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
    std::vector<float> weights(battle_common::kBattleOutputDim, 1.0f);
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

Metrics evaluate_validation(Model& model,
                            BattleBatchLoader& loader,
                            const TrainConfig& train_config,
                            const std::vector<float>& class_weights) {
    model.set_training(false);
    loader.reset();

    Metrics m;
    Tensor inputs({1, battle_common::kBattleInputDim}, 0.0f);
    std::vector<uint32_t> targets;
    while (loader.next(inputs, targets)) {
        for (size_t i = 0; i < targets.size(); ++i) {
            Tensor sample({1, battle_common::kBattleInputDim}, 0.0f);
            const float* src = &inputs.data[i * battle_common::kBattleInputDim];
            std::copy(src, src + battle_common::kBattleInputDim, sample.data.begin());

            Tensor logits = model.forward(sample);
            const uint32_t target = targets[i];
            const float weight = target < class_weights.size() ? class_weights[target] : 1.0f;
            m.loss += cross_entropy_advanced(logits, target, train_config.label_smoothing, weight);
            m.top1 += top_k_hit(logits, target, 1) ? 1.0f : 0.0f;
            m.top3 += top_k_hit(logits, target, 3) ? 1.0f : 0.0f;
            m.top5 += top_k_hit(logits, target, 5) ? 1.0f : 0.0f;
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
    features.reserve(samples.size());
    actions.reserve(samples.size());
    for (const BattleSample& s : samples) {
        features.push_back(s.features);
        actions.push_back(s.action);
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

    BattleBatchLoader train_loader(&features, &actions, train_indices, train_config.batch_size, true, 1234);
    BattleBatchLoader val_loader(&features, &actions, val_indices, train_config.batch_size, false, 5678);
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
        while (train_loader.next(batch_inputs, targets)) {
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

#pragma omp parallel default(none) shared(batch, batch_inputs, targets, class_weights, train_config, worker_models, thread_grad_sums, thread_loss, thread_top1, thread_top3, thread_top5, thread_count)
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
                    const uint32_t target = targets[i];
                    const float weight = target < class_weights.size() ? class_weights[target] : 1.0f;

                    thread_loss[static_cast<size_t>(tid)] +=
                        cross_entropy_advanced(logits, target, train_config.label_smoothing, weight);
                    thread_top1[static_cast<size_t>(tid)] += top_k_hit(logits, target, 1) ? 1.0f : 0.0f;
                    thread_top3[static_cast<size_t>(tid)] += top_k_hit(logits, target, 3) ? 1.0f : 0.0f;
                    thread_top5[static_cast<size_t>(tid)] += top_k_hit(logits, target, 5) ? 1.0f : 0.0f;
                    thread_count[static_cast<size_t>(tid)] += 1;

                    Tensor grad = grad_cross_entropy_advanced(logits, target, train_config.label_smoothing, weight);
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
    return 0;
}
