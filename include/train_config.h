#ifndef TRAIN_CONFIG_H
#define TRAIN_CONFIG_H

#include <cstddef>
#include <string>

struct TrainConfig {
    size_t epochs = 10;
    size_t batch_size = 32;
    float validation_split = 0.1f;
    size_t early_stopping_patience = 5;

    float base_lr = 0.001f;
    std::string optimizer = "adam";
    float weight_decay = 1e-4f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;

    std::string scheduler = "cosine";
    size_t step_size = 5;
    float gamma = 0.9f;
    float min_lr = 1e-5f;

    float label_smoothing = 0.0f;
    bool use_class_weights = false;
    std::string class_weights_path = "../data/class_weights.txt";
};

TrainConfig load_train_config(const std::string& config_path);
float scheduler_lr(const TrainConfig& cfg, size_t epoch_index);
bool validate_train_config(const TrainConfig& cfg, std::string& error_message);

#endif
