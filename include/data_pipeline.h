#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

#include "battle_common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct BattleDatasetConfig {
    std::string data_root = "../data";
    std::string json_path = "../data/battle_train.json";
    std::string txt_path = "../data/battle_train.txt";
    std::string vocab_path = "../data/vocab.txt";
    uint32_t rng_seed = 20260305;
    size_t synthetic_samples = 20000;
    size_t cluster_count = 10;
    size_t sequence_length = 10;
    float edge_case_rate = 0.045f;
    float validation_tolerance = 0.35f;
};

struct BattleSample {
    std::array<float, battle_common::kBattleInputDim> features{};
    uint32_t action = 0;
};

struct BattleDatasetInfo {
    size_t sample_count = 0;
    bool rebuilt = false;
    std::string json_path;
    std::string txt_path;
    std::string vocab_path;
};

BattleDatasetInfo prepare_battle_dataset(const BattleDatasetConfig& config,
                                         std::vector<BattleSample>& samples,
                                         bool force_rebuild);

BattleDatasetConfig make_battle_dataset_config(const std::string& data_root);

#endif
