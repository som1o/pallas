#include "train_config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace {

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

}  // namespace

TrainConfig load_train_config(const std::string& config_path) {
    TrainConfig cfg;
    std::ifstream file(config_path);
    if (!file) {
        return cfg;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();
    std::string t = trim(content);

    if (!t.empty() && t[0] == '{') {
        try {
            const auto j = nlohmann::json::parse(content);
            if (j.contains("epochs")) cfg.epochs = j["epochs"].get<size_t>();
            if (j.contains("batch_size")) cfg.batch_size = j["batch_size"].get<size_t>();
            if (j.contains("validation_split")) cfg.validation_split = j["validation_split"].get<float>();
            if (j.contains("early_stopping_patience")) cfg.early_stopping_patience = j["early_stopping_patience"].get<size_t>();
            if (j.contains("base_lr")) cfg.base_lr = j["base_lr"].get<float>();
            if (j.contains("optimizer")) cfg.optimizer = j["optimizer"].get<std::string>();
            if (j.contains("weight_decay")) cfg.weight_decay = j["weight_decay"].get<float>();
            if (j.contains("adam_beta1")) cfg.adam_beta1 = j["adam_beta1"].get<float>();
            if (j.contains("adam_beta2")) cfg.adam_beta2 = j["adam_beta2"].get<float>();
            if (j.contains("adam_epsilon")) cfg.adam_epsilon = j["adam_epsilon"].get<float>();
            if (j.contains("scheduler")) cfg.scheduler = j["scheduler"].get<std::string>();
            if (j.contains("step_size")) cfg.step_size = j["step_size"].get<size_t>();
            if (j.contains("gamma")) cfg.gamma = j["gamma"].get<float>();
            if (j.contains("min_lr")) cfg.min_lr = j["min_lr"].get<float>();
            if (j.contains("label_smoothing")) cfg.label_smoothing = j["label_smoothing"].get<float>();
            if (j.contains("use_class_weights")) cfg.use_class_weights = j["use_class_weights"].get<bool>();
            if (j.contains("class_weights_path")) cfg.class_weights_path = j["class_weights_path"].get<std::string>();
            if (j.contains("use_actor_critic")) cfg.use_actor_critic = j["use_actor_critic"].get<bool>();
            if (j.contains("policy_loss_weight")) cfg.policy_loss_weight = j["policy_loss_weight"].get<float>();
            if (j.contains("value_loss_weight")) cfg.value_loss_weight = j["value_loss_weight"].get<float>();
            if (j.contains("reward_scale")) cfg.reward_scale = j["reward_scale"].get<float>();
            if (j.contains("entropy_coeff")) cfg.entropy_coeff = j["entropy_coeff"].get<float>();
        } catch (...) {
            return cfg;
        }
    } else {
        std::stringstream lines(content);
        std::string line;
        while (std::getline(lines, line)) {
            std::string s = trim(line);
            if (s.empty() || s[0] == '#') continue;
            auto parse_value = [&](const std::string& key) -> std::string {
                return trim(s.substr(key.size()));
            };
            try {
                if (s.rfind("epochs:", 0) == 0) cfg.epochs = static_cast<size_t>(std::stoul(parse_value("epochs:")));
                else if (s.rfind("batch_size:", 0) == 0) cfg.batch_size = static_cast<size_t>(std::stoul(parse_value("batch_size:")));
                else if (s.rfind("validation_split:", 0) == 0) cfg.validation_split = std::stof(parse_value("validation_split:"));
                else if (s.rfind("early_stopping_patience:", 0) == 0) cfg.early_stopping_patience = static_cast<size_t>(std::stoul(parse_value("early_stopping_patience:")));
                else if (s.rfind("base_lr:", 0) == 0) cfg.base_lr = std::stof(parse_value("base_lr:"));
                else if (s.rfind("optimizer:", 0) == 0) cfg.optimizer = parse_value("optimizer:");
                else if (s.rfind("weight_decay:", 0) == 0) cfg.weight_decay = std::stof(parse_value("weight_decay:"));
                else if (s.rfind("adam_beta1:", 0) == 0) cfg.adam_beta1 = std::stof(parse_value("adam_beta1:"));
                else if (s.rfind("adam_beta2:", 0) == 0) cfg.adam_beta2 = std::stof(parse_value("adam_beta2:"));
                else if (s.rfind("adam_epsilon:", 0) == 0) cfg.adam_epsilon = std::stof(parse_value("adam_epsilon:"));
                else if (s.rfind("scheduler:", 0) == 0) cfg.scheduler = parse_value("scheduler:");
                else if (s.rfind("step_size:", 0) == 0) cfg.step_size = static_cast<size_t>(std::stoul(parse_value("step_size:")));
                else if (s.rfind("gamma:", 0) == 0) cfg.gamma = std::stof(parse_value("gamma:"));
                else if (s.rfind("min_lr:", 0) == 0) cfg.min_lr = std::stof(parse_value("min_lr:"));
                else if (s.rfind("label_smoothing:", 0) == 0) cfg.label_smoothing = std::stof(parse_value("label_smoothing:"));
                else if (s.rfind("use_class_weights:", 0) == 0) {
                    const std::string v = parse_value("use_class_weights:");
                    cfg.use_class_weights = (v == "true" || v == "1" || v == "yes");
                } else if (s.rfind("class_weights_path:", 0) == 0) {
                    cfg.class_weights_path = parse_value("class_weights_path:");
                } else if (s.rfind("use_actor_critic:", 0) == 0) {
                    const std::string v = parse_value("use_actor_critic:");
                    cfg.use_actor_critic = (v == "true" || v == "1" || v == "yes");
                } else if (s.rfind("policy_loss_weight:", 0) == 0) {
                    cfg.policy_loss_weight = std::stof(parse_value("policy_loss_weight:"));
                } else if (s.rfind("value_loss_weight:", 0) == 0) {
                    cfg.value_loss_weight = std::stof(parse_value("value_loss_weight:"));
                } else if (s.rfind("reward_scale:", 0) == 0) {
                    cfg.reward_scale = std::stof(parse_value("reward_scale:"));
                } else if (s.rfind("entropy_coeff:", 0) == 0) {
                    cfg.entropy_coeff = std::stof(parse_value("entropy_coeff:"));
                }
            } catch (...) {
            }
        }
    }

    cfg.validation_split = std::min(0.5f, std::max(0.01f, cfg.validation_split));
    cfg.label_smoothing = std::min(0.999f, std::max(0.0f, cfg.label_smoothing));
    cfg.early_stopping_patience = std::max<size_t>(1, cfg.early_stopping_patience);
    return cfg;
}

float scheduler_lr(const TrainConfig& cfg, size_t epoch_index) {
    if (cfg.scheduler == "step") {
        size_t step = cfg.step_size == 0 ? 1 : cfg.step_size;
        size_t k = epoch_index / step;
        return cfg.base_lr * std::pow(cfg.gamma, static_cast<float>(k));
    }
    if (cfg.scheduler == "exponential") {
        return cfg.base_lr * std::pow(cfg.gamma, static_cast<float>(epoch_index));
    }
    if (cfg.scheduler == "cosine") {
        constexpr float pi = 3.14159265358979323846f;
        const float t = static_cast<float>(epoch_index);
        const float T = static_cast<float>(std::max<size_t>(1, cfg.epochs - 1));
        const float cosine = 0.5f * (1.0f + std::cos(pi * t / T));
        return cfg.min_lr + (cfg.base_lr - cfg.min_lr) * cosine;
    }
    return cfg.base_lr;
}

bool validate_train_config(const TrainConfig& cfg, std::string& error_message) {
    if (cfg.epochs == 0) {
        error_message = "epochs must be > 0";
        return false;
    }
    if (cfg.batch_size == 0) {
        error_message = "batch_size must be > 0";
        return false;
    }
    if (cfg.base_lr <= 0.0f) {
        error_message = "base_lr must be > 0";
        return false;
    }
    if (cfg.min_lr < 0.0f || cfg.min_lr > cfg.base_lr) {
        error_message = "min_lr must be in [0, base_lr]";
        return false;
    }
    if (cfg.validation_split < 0.01f || cfg.validation_split > 0.5f) {
        error_message = "validation_split must be in [0.01, 0.5]";
        return false;
    }
    if (cfg.early_stopping_patience == 0) {
        error_message = "early_stopping_patience must be > 0";
        return false;
    }
    if (cfg.label_smoothing < 0.0f || cfg.label_smoothing >= 1.0f) {
        error_message = "label_smoothing must be in [0, 1)";
        return false;
    }
    if (cfg.optimizer != "adam" && cfg.optimizer != "sgd") {
        error_message = "optimizer must be 'adam' or 'sgd'";
        return false;
    }
    if (cfg.scheduler != "step" && cfg.scheduler != "exponential" && cfg.scheduler != "cosine") {
        error_message = "scheduler must be one of: step, exponential, cosine";
        return false;
    }
    if (cfg.weight_decay < 0.0f) {
        error_message = "weight_decay must be >= 0";
        return false;
    }
    if (cfg.policy_loss_weight <= 0.0f) {
        error_message = "policy_loss_weight must be > 0";
        return false;
    }
    if (cfg.value_loss_weight < 0.0f) {
        error_message = "value_loss_weight must be >= 0";
        return false;
    }
    if (cfg.reward_scale <= 0.0f) {
        error_message = "reward_scale must be > 0";
        return false;
    }
    if (cfg.entropy_coeff < 0.0f) {
        error_message = "entropy_coeff must be >= 0";
        return false;
    }
    return true;
}
