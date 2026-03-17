#include "model.h"

#include "common_utils.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string_view>
#include <stdexcept>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#define PALLAS_HAS_NLOHMANN_JSON 1
#else
#define PALLAS_HAS_NLOHMANN_JSON 0
#endif

namespace {

std::vector<size_t> parse_number_list(std::string_view raw) {
    std::vector<size_t> out;
    size_t i = 0;
    while (i < raw.size()) {
        while (i < raw.size() &&
               (std::isspace(static_cast<unsigned char>(raw[i])) || raw[i] == '[' || raw[i] == ']' || raw[i] == ',')) {
            ++i;
        }
        if (i >= raw.size()) {
            break;
        }
        size_t j = i;
        while (j < raw.size() && std::isdigit(static_cast<unsigned char>(raw[j]))) {
            ++j;
        }
        if (j == i) {
            ++i;
            continue;
        }

        size_t value = 0;
        const char* begin = raw.data() + i;
        const char* end = raw.data() + j;
        const auto parsed = std::from_chars(begin, end, value);
        if (parsed.ec == std::errc()) {
            out.push_back(value);
        }
        i = j;
    }
    return out;
}

std::string parse_json_string_field(const std::string& content, const std::string& key, const std::string& fallback) {
    const std::string_view view(content);
    const std::string needle = "\"" + key + "\"";
    size_t p = view.find(needle);
    if (p == std::string::npos) return fallback;
    p = view.find(':', p);
    if (p == std::string::npos) return fallback;
    size_t q1 = view.find('"', p + 1);
    if (q1 == std::string::npos) return fallback;
    size_t q2 = view.find('"', q1 + 1);
    if (q2 == std::string::npos) return fallback;
    return std::string(view.substr(q1 + 1, q2 - q1 - 1));
}

float parse_json_float_field(const std::string& content, const std::string& key, float fallback) {
    const std::string_view view(content);
    const std::string needle = "\"" + key + "\"";
    size_t p = view.find(needle);
    if (p == std::string::npos) return fallback;
    p = view.find(':', p);
    if (p == std::string::npos) return fallback;
    size_t start = view.find_first_of("-0123456789.", p + 1);
    if (start == std::string::npos) return fallback;
    size_t end = start;
    while (end < view.size() && (std::isdigit(static_cast<unsigned char>(view[end])) || view[end] == '.' || view[end] == '-')) {
        ++end;
    }
    try {
        return std::stof(std::string(view.substr(start, end - start)));
    } catch (...) {
        return fallback;
    }
}

bool parse_json_bool_field(const std::string& content, const std::string& key, bool fallback) {
    const std::string_view view(content);
    const std::string needle = "\"" + key + "\"";
    size_t p = view.find(needle);
    if (p == std::string::npos) return fallback;
    p = view.find(':', p);
    if (p == std::string::npos) return fallback;
    const size_t t = view.find("true", p + 1);
    const size_t f = view.find("false", p + 1);
    if (t != std::string::npos && (f == std::string::npos || t < f)) return true;
    if (f != std::string::npos && (t == std::string::npos || f < t)) return false;
    return fallback;
}

std::vector<size_t> parse_json_hidden(const std::string& content) {
    const std::string_view view(content);
    const std::string needle = "\"hidden_layers\"";
    size_t p = view.find(needle);
    if (p == std::string::npos) return {};
    size_t l = view.find('[', p);
    size_t r = view.find(']', l);
    if (l == std::string::npos || r == std::string::npos) return {};
    return parse_number_list(view.substr(l, r - l + 1));
}

class ReLUActivation final : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor out(input.shape, 0.0f);
        for (size_t i = 0; i < input.data.size(); ++i) {
            out.data[i] = std::max(0.0f, input.data[i]);
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output, const Tensor& pre_activation) const override {
        Tensor grad(pre_activation.shape, 0.0f);
        for (size_t i = 0; i < pre_activation.data.size(); ++i) {
            grad.data[i] = pre_activation.data[i] > 0.0f ? grad_output.data[i] : 0.0f;
        }
        return grad;
    }
};

class SigmoidActivation final : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor out(input.shape, 0.0f);
        for (size_t i = 0; i < input.data.size(); ++i) {
            out.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output, const Tensor& pre_activation) const override {
        Tensor grad(pre_activation.shape, 0.0f);
        for (size_t i = 0; i < pre_activation.data.size(); ++i) {
            float s = 1.0f / (1.0f + std::exp(-pre_activation.data[i]));
            grad.data[i] = grad_output.data[i] * s * (1.0f - s);
        }
        return grad;
    }
};

class TanhActivation final : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor out(input.shape, 0.0f);
        for (size_t i = 0; i < input.data.size(); ++i) {
            out.data[i] = std::tanh(input.data[i]);
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output, const Tensor& pre_activation) const override {
        Tensor grad(pre_activation.shape, 0.0f);
        for (size_t i = 0; i < pre_activation.data.size(); ++i) {
            float t = std::tanh(pre_activation.data[i]);
            grad.data[i] = grad_output.data[i] * (1.0f - t * t);
        }
        return grad;
    }
};

class LeakyReLUActivation final : public Activation {
public:
    explicit LeakyReLUActivation(float alpha) : alpha_(alpha) {}

    Tensor forward(const Tensor& input) const override {
        Tensor out(input.shape, 0.0f);
        for (size_t i = 0; i < input.data.size(); ++i) {
            out.data[i] = input.data[i] > 0.0f ? input.data[i] : alpha_ * input.data[i];
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output, const Tensor& pre_activation) const override {
        Tensor grad(pre_activation.shape, 0.0f);
        for (size_t i = 0; i < pre_activation.data.size(); ++i) {
            float slope = pre_activation.data[i] > 0.0f ? 1.0f : alpha_;
            grad.data[i] = grad_output.data[i] * slope;
        }
        return grad;
    }

private:
    float alpha_;
};

std::unique_ptr<Activation> build_activation(const ModelConfig& config) {
    if (config.activation == "relu") {
        return std::make_unique<ReLUActivation>();
    }
    if (config.activation == "sigmoid") {
        return std::make_unique<SigmoidActivation>();
    }
    if (config.activation == "tanh") {
        return std::make_unique<TanhActivation>();
    }
    if (config.activation == "leaky_relu") {
        return std::make_unique<LeakyReLUActivation>(config.leaky_relu_alpha);
    }
    return std::make_unique<ReLUActivation>();
}

constexpr uint32_t kModelMagic = 0x4D4F4458;
constexpr uint32_t kModelStateVersion = 2;

std::string_view trim_view(std::string_view text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool parse_size_t_value(std::string_view text, size_t* out) {
    if (out == nullptr || text.empty()) {
        return false;
    }
    size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        return false;
    }
    *out = value;
    return true;
}

void hash_mix_u64(uint64_t& h, uint64_t value) {
    h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
}

uint64_t architecture_hash(size_t input_dim, size_t output_dim, const ModelConfig& model_config) {
    uint64_t hash = 1469598103934665603ULL;
    hash_mix_u64(hash, static_cast<uint64_t>(input_dim));
    hash_mix_u64(hash, static_cast<uint64_t>(output_dim));
    for (size_t hidden : model_config.hidden_layers) {
        hash_mix_u64(hash, static_cast<uint64_t>(hidden));
    }
    for (char ch : model_config.activation) {
        hash_mix_u64(hash, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
    }
    for (char ch : model_config.norm) {
        hash_mix_u64(hash, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
    }
    hash_mix_u64(hash, model_config.use_dropout ? 1ULL : 0ULL);
    hash_mix_u64(hash, static_cast<uint64_t>(std::llround(model_config.dropout_prob * 1000000.0f)));
    hash_mix_u64(hash, static_cast<uint64_t>(std::llround(model_config.leaky_relu_alpha * 1000000.0f)));
    return hash;
}

template <typename T>
bool read_binary(std::ifstream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

template <typename T>
void write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

std::string read_string(std::ifstream& in) {
    size_t len = 0;
    if (!read_binary(in, &len)) {
        return {};
    }
    std::string value(len, '\0');
    if (len > 0) {
        in.read(value.data(), static_cast<std::streamsize>(len));
        if (!in) {
            return {};
        }
    }
    return value;
}

void write_string(std::ofstream& out, const std::string& value) {
    const size_t len = value.size();
    write_binary(out, len);
    out.write(value.data(), static_cast<std::streamsize>(len));
}

}  // namespace

ModelConfig load_model_config(const std::string& config_path) {
    ModelConfig cfg;
    cfg.hidden_layers = {512, 384, 256, 192};

    std::ifstream file(config_path);
    if (!file) {
        return cfg;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();
    std::string t = pallas::util::trim_copy(content);

    if (!t.empty() && t[0] == '{') {
#if PALLAS_HAS_NLOHMANN_JSON
        try {
            const auto j = nlohmann::json::parse(content);
            if (j.contains("hidden_layers") && j["hidden_layers"].is_array()) {
                cfg.hidden_layers.clear();
                for (const auto& item : j["hidden_layers"]) {
                    if (item.is_number_unsigned()) {
                        cfg.hidden_layers.push_back(item.get<size_t>());
                    }
                }
            }
            if (j.contains("activation") && j["activation"].is_string()) {
                cfg.activation = j["activation"].get<std::string>();
            }
            if (j.contains("norm") && j["norm"].is_string()) {
                cfg.norm = j["norm"].get<std::string>();
            }
            if (j.contains("dropout_prob") && j["dropout_prob"].is_number()) {
                cfg.dropout_prob = j["dropout_prob"].get<float>();
            }
            if (j.contains("use_dropout") && j["use_dropout"].is_boolean()) {
                cfg.use_dropout = j["use_dropout"].get<bool>();
            }
            if (j.contains("leaky_relu_alpha") && j["leaky_relu_alpha"].is_number()) {
                cfg.leaky_relu_alpha = j["leaky_relu_alpha"].get<float>();
            }
            if (cfg.hidden_layers.empty()) {
                cfg.hidden_layers = {512, 384, 256, 192};
            }
            return cfg;
        } catch (...) {
        }
#endif
        std::vector<size_t> layers = parse_json_hidden(content);
        if (!layers.empty()) {
            cfg.hidden_layers = layers;
        }
        cfg.activation = parse_json_string_field(content, "activation", cfg.activation);
        cfg.norm = parse_json_string_field(content, "norm", cfg.norm);
        cfg.dropout_prob = parse_json_float_field(content, "dropout_prob", cfg.dropout_prob);
        cfg.use_dropout = parse_json_bool_field(content, "use_dropout", cfg.use_dropout);
        cfg.leaky_relu_alpha = parse_json_float_field(content, "leaky_relu_alpha", cfg.leaky_relu_alpha);
        return cfg;
    }

    std::stringstream lines(content);
    std::string line;
    bool reading_hidden = false;
    constexpr std::string_view kHiddenPrefix = "hidden_layers:";
    constexpr std::string_view kActivationPrefix = "activation:";
    constexpr std::string_view kNormPrefix = "norm:";
    constexpr std::string_view kDropoutPrefix = "dropout_prob:";
    constexpr std::string_view kUseDropoutPrefix = "use_dropout:";
    constexpr std::string_view kLeakyAlphaPrefix = "leaky_relu_alpha:";
    while (std::getline(lines, line)) {
        const std::string_view s = trim_view(line);
        if (s.empty() || s[0] == '#') {
            continue;
        }
        if (s.rfind(kHiddenPrefix, 0) == 0) {
            std::string_view rhs = trim_view(s.substr(kHiddenPrefix.size()));
            if (!rhs.empty()) {
                std::vector<size_t> layers = parse_number_list(rhs);
                if (!layers.empty()) cfg.hidden_layers = layers;
                reading_hidden = false;
            } else {
                cfg.hidden_layers.clear();
                reading_hidden = true;
            }
            continue;
        }
        if (reading_hidden && s.rfind("-", 0) == 0) {
            size_t layer = 0;
            if (parse_size_t_value(trim_view(s.substr(1)), &layer)) {
                cfg.hidden_layers.push_back(layer);
            }
            continue;
        }
        reading_hidden = false;
        if (s.rfind(kActivationPrefix, 0) == 0) {
            cfg.activation = std::string(trim_view(s.substr(kActivationPrefix.size())));
        } else if (s.rfind(kNormPrefix, 0) == 0) {
            cfg.norm = std::string(trim_view(s.substr(kNormPrefix.size())));
        } else if (s.rfind(kDropoutPrefix, 0) == 0) {
            try { cfg.dropout_prob = std::stof(std::string(trim_view(s.substr(kDropoutPrefix.size())))); } catch (...) {}
        } else if (s.rfind(kUseDropoutPrefix, 0) == 0) {
            std::string b = std::string(trim_view(s.substr(kUseDropoutPrefix.size())));
            cfg.use_dropout = (b == "true" || b == "1" || b == "yes");
        } else if (s.rfind(kLeakyAlphaPrefix, 0) == 0) {
            try { cfg.leaky_relu_alpha = std::stof(std::string(trim_view(s.substr(kLeakyAlphaPrefix.size())))); } catch (...) {}
        }
    }

    if (cfg.hidden_layers.empty()) {
        cfg.hidden_layers = {512, 384, 256, 192};
    }
    return cfg;
}

bool validate_model_config(const ModelConfig& config, std::string& error_message) {
    if (config.hidden_layers.empty()) {
        error_message = "hidden_layers must contain at least one entry";
        return false;
    }
    for (size_t hidden : config.hidden_layers) {
        if (hidden == 0) {
            error_message = "hidden layer size must be > 0";
            return false;
        }
    }
    if (config.activation != "relu" &&
        config.activation != "sigmoid" &&
        config.activation != "tanh" &&
        config.activation != "leaky_relu") {
        error_message = "activation must be one of: relu, sigmoid, tanh, leaky_relu";
        return false;
    }
    if (config.norm != "layernorm" && config.norm != "batchnorm" && config.norm != "none") {
        error_message = "norm must be one of: layernorm, batchnorm, none";
        return false;
    }
    if (config.dropout_prob < 0.0f || config.dropout_prob >= 1.0f) {
        error_message = "dropout_prob must be in [0, 1)";
        return false;
    }
    if (config.leaky_relu_alpha < 0.0f) {
        error_message = "leaky_relu_alpha must be >= 0";
        return false;
    }
    return true;
}

ModelStateInspection inspect_model_state(const std::string& path) {
    ModelStateInspection inspection;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        inspection.error_message = "inspect_model_state: failed to open file";
        return inspection;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t arch_hash = 0;
    if (!read_binary(in, &magic) || !read_binary(in, &version) || !read_binary(in, &arch_hash)) {
        inspection.error_message = "inspect_model_state: invalid header";
        return inspection;
    }
    if (magic != kModelMagic || version != kModelStateVersion) {
        inspection.error_message = "inspect_model_state: unsupported model state format";
        return inspection;
    }

    uint64_t timestamp_unix = 0;
    size_t epoch = 0;
    float val_loss = 0.0f;
    float val_top1 = 0.0f;
    if (!read_binary(in, &timestamp_unix) ||
        !read_binary(in, &epoch) ||
        !read_binary(in, &val_loss) ||
        !read_binary(in, &val_top1)) {
        inspection.error_message = "inspect_model_state: invalid metadata";
        return inspection;
    }
    (void)read_string(in);

    size_t in_dim = 0;
    size_t out_dim = 0;
    if (!read_binary(in, &in_dim) || !read_binary(in, &out_dim)) {
        inspection.error_message = "inspect_model_state: missing dimensions";
        return inspection;
    }

    size_t hidden_count = 0;
    if (!read_binary(in, &hidden_count)) {
        inspection.error_message = "inspect_model_state: missing hidden layer count";
        return inspection;
    }

    ModelConfig cfg;
    cfg.hidden_layers.assign(hidden_count, 0);
    for (size_t i = 0; i < hidden_count; ++i) {
        if (!read_binary(in, &cfg.hidden_layers[i])) {
            inspection.error_message = "inspect_model_state: invalid hidden layer data";
            return inspection;
        }
    }

    cfg.activation = read_string(in);
    cfg.norm = read_string(in);
    if (!read_binary(in, &cfg.use_dropout) ||
        !read_binary(in, &cfg.dropout_prob) ||
        !read_binary(in, &cfg.leaky_relu_alpha)) {
        inspection.error_message = "inspect_model_state: invalid model config payload";
        return inspection;
    }

    inspection.ok = true;
    inspection.input_dim = in_dim;
    inspection.output_dim = out_dim;
    inspection.model_config = std::move(cfg);
    return inspection;
}

Model::Model(size_t input_dim, size_t output_dim, const ModelConfig& config)
    : norm_type_(config.norm),
      use_norm_(config.norm == "layernorm" || config.norm == "batchnorm"),
      use_dropout_(config.use_dropout),
      dropout_prob_(std::min(0.9f, std::max(0.0f, config.dropout_prob))),
      training_(true),
    inference_only_(false),
            eps_(1e-5f),
            adam_t_(0),
            dropout_rng_(std::random_device{}()),
            dropout_dist_(0.0f, 1.0f) {
    std::vector<size_t> dims;
    dims.push_back(input_dim);
    for (size_t h : config.hidden_layers) {
        dims.push_back(h);
    }
    dims.push_back(output_dim);

    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        layers_.emplace_back(dims[i], dims[i + 1]);
    }

    const size_t hidden_count = layers_.size() > 0 ? layers_.size() - 1 : 0;
    activations_.reserve(hidden_count);
    gamma_.reserve(hidden_count);
    beta_.reserve(hidden_count);
    for (size_t i = 0; i < hidden_count; ++i) {
        activations_.push_back(build_activation(config));
        gamma_.emplace_back(std::vector<size_t>{layers_[i].out_dim}, 1.0f);
        beta_.emplace_back(std::vector<size_t>{layers_[i].out_dim}, 0.0f);
    }

    const size_t gsize = gradient_size();
    adam_m_.assign(gsize, 0.0f);
    adam_v_.assign(gsize, 0.0f);
}

void Model::configure_optimizer(const OptimizerConfig& config) {
    optimizer_config_ = config;
    adam_t_ = 0;
    const size_t gsize = gradient_size();
    adam_m_.assign(gsize, 0.0f);
    adam_v_.assign(gsize, 0.0f);
}

Tensor Model::apply_norm_forward(const Tensor& x, size_t layer_idx) {
    const size_t batch = x.shape[0];
    const size_t features = x.shape[1];

    Tensor xhat(x.shape, 0.0f);
    Tensor out(x.shape, 0.0f);
    Tensor var_cache({batch, features}, 0.0f);

    // LayerNorm path: normalize each sample independently over its feature axis.
    // This keeps behavior stable for variable batch sizes and inference-time single-item calls.
    if (norm_type_ == "layernorm") {
        for (size_t b = 0; b < batch; ++b) {
            float mean = 0.0f;
            for (size_t f = 0; f < features; ++f) {
                mean += x.data[b * features + f];
            }
            mean /= static_cast<float>(features);

            float var = 0.0f;
            for (size_t f = 0; f < features; ++f) {
                float d = x.data[b * features + f] - mean;
                var += d * d;
            }
            var /= static_cast<float>(features);
            float inv_std = 1.0f / std::sqrt(var + eps_);

            for (size_t f = 0; f < features; ++f) {
                size_t idx = b * features + f;
                xhat.data[idx] = (x.data[idx] - mean) * inv_std;
                var_cache.data[idx] = var;
                out.data[idx] = gamma_[layer_idx].data[f] * xhat.data[idx] + beta_[layer_idx].data[f];
            }
        }
    } else {
        // BatchNorm path: compute per-feature moments over the current batch.
        // We cache variance per feature for the corresponding backward transform.
        std::vector<float> mean(features, 0.0f), var(features, 0.0f);
        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch; ++b) {
                mean[f] += x.data[b * features + f];
            }
            mean[f] /= static_cast<float>(batch);
            for (size_t b = 0; b < batch; ++b) {
                float d = x.data[b * features + f] - mean[f];
                var[f] += d * d;
            }
            var[f] /= static_cast<float>(batch);
        }

        for (size_t b = 0; b < batch; ++b) {
            for (size_t f = 0; f < features; ++f) {
                size_t idx = b * features + f;
                float inv_std = 1.0f / std::sqrt(var[f] + eps_);
                xhat.data[idx] = (x.data[idx] - mean[f]) * inv_std;
                var_cache.data[idx] = var[f];
                out.data[idx] = gamma_[layer_idx].data[f] * xhat.data[idx] + beta_[layer_idx].data[f];
            }
        }
    }

    if (norm_xhat_cache_.size() <= layer_idx) {
        norm_xhat_cache_.resize(layer_idx + 1, Tensor({1, 1}, 0.0f));
        norm_var_cache_.resize(layer_idx + 1, Tensor({1, 1}, 0.0f));
    }
    norm_xhat_cache_[layer_idx] = xhat;
    norm_var_cache_[layer_idx] = var_cache;
    return out;
}

Tensor Model::apply_norm_backward(const Tensor& grad_output, size_t layer_idx) {
    const Tensor& xhat = norm_xhat_cache_[layer_idx];
    const Tensor& var_cache = norm_var_cache_[layer_idx];
    const size_t batch = grad_output.shape[0];
    const size_t features = grad_output.shape[1];

    Tensor grad_input(grad_output.shape, 0.0f);
    std::vector<float> dgamma(features, 0.0f);
    std::vector<float> dbeta(features, 0.0f);
    std::vector<float> inv_std(features, 0.0f);

    for (size_t f = 0; f < features; ++f) {
        inv_std[f] = 1.0f / std::sqrt(var_cache.data[f] + eps_);
        for (size_t b = 0; b < batch; ++b) {
            size_t idx = b * features + f;
            dgamma[f] += grad_output.data[idx] * xhat.data[idx];
            dbeta[f] += grad_output.data[idx];
        }
    }

    for (size_t f = 0; f < features; ++f) {
        gamma_[layer_idx].grad[f] += dgamma[f];
        beta_[layer_idx].grad[f] += dbeta[f];
    }

    // LayerNorm backward: apply closed-form normalized gradient for each sample.
    if (norm_type_ == "layernorm") {
        for (size_t b = 0; b < batch; ++b) {
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (size_t f = 0; f < features; ++f) {
                size_t idx = b * features + f;
                float dxhat = grad_output.data[idx] * gamma_[layer_idx].data[f];
                sum1 += dxhat;
                sum2 += dxhat * xhat.data[idx];
            }
            for (size_t f = 0; f < features; ++f) {
                size_t idx = b * features + f;
                float dxhat = grad_output.data[idx] * gamma_[layer_idx].data[f];
                float inv = 1.0f / std::sqrt(var_cache.data[idx] + eps_);
                grad_input.data[idx] = inv * (dxhat - sum1 / static_cast<float>(features)
                    - xhat.data[idx] * sum2 / static_cast<float>(features));
            }
        }
    } else {
        // BatchNorm backward: aggregate feature-wise reductions over batch, then project back.
        for (size_t f = 0; f < features; ++f) {
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (size_t b = 0; b < batch; ++b) {
                size_t idx = b * features + f;
                float dxhat = grad_output.data[idx] * gamma_[layer_idx].data[f];
                sum1 += dxhat;
                sum2 += dxhat * xhat.data[idx];
            }
            for (size_t b = 0; b < batch; ++b) {
                size_t idx = b * features + f;
                float dxhat = grad_output.data[idx] * gamma_[layer_idx].data[f];
                grad_input.data[idx] = inv_std[f] * (dxhat - sum1 / static_cast<float>(batch)
                    - xhat.data[idx] * sum2 / static_cast<float>(batch));
            }
        }
    }

    return grad_input;
}

Tensor Model::forward(const Tensor& input) {
    if (!inference_only_) {
        layer_inputs_.clear();
        norm_outputs_.clear();
        dropout_masks_.clear();
        norm_xhat_cache_.clear();
        norm_var_cache_.clear();
    }

    Tensor current = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        if (!inference_only_) {
            layer_inputs_.push_back(current);
        }
        Tensor z = layers_[i].forward(current);

        if (i + 1 == layers_.size()) {
            current = z;
            continue;
        }

        Tensor normed = use_norm_ ? apply_norm_forward(z, i) : z;
        if (!inference_only_) {
            norm_outputs_.push_back(normed);
        }

        Tensor activated = activations_[i]->forward(normed);

        if (use_dropout_ && training_ && dropout_prob_ > 0.0f) {
            Tensor mask(activated.shape, 0.0f);
            const float keep = 1.0f - dropout_prob_;
            for (size_t idx = 0; idx < activated.data.size(); ++idx) {
                mask.data[idx] = (dropout_dist_(dropout_rng_) < keep) ? 1.0f : 0.0f;
                activated.data[idx] = activated.data[idx] * mask.data[idx] / keep;
            }
            dropout_masks_.push_back(mask);
        } else if (!inference_only_) {
            dropout_masks_.push_back(Tensor(activated.shape, 1.0f));
        }

        current = activated;
    }
    return current;
}

void Model::backward(const Tensor& grad_output) {
    if (inference_only_) {
        return;
    }
    Tensor grad = grad_output;

    for (int li = static_cast<int>(layers_.size()) - 1; li >= 0; --li) {
        const size_t i = static_cast<size_t>(li);

        if (i + 1 != layers_.size()) {
            if (use_dropout_ && training_ && dropout_prob_ > 0.0f) {
                const float keep = 1.0f - dropout_prob_;
                for (size_t idx = 0; idx < grad.data.size(); ++idx) {
                    grad.data[idx] = grad.data[idx] * dropout_masks_[i].data[idx] / keep;
                }
            }

            grad = activations_[i]->backward(grad, norm_outputs_[i]);
            if (use_norm_) {
                grad = apply_norm_backward(grad, i);
            }
        }

        grad = layers_[i].backward(grad, layer_inputs_[i]);
    }
}

void Model::update(float lr) {
    if (inference_only_) {
        return;
    }
    if (optimizer_config_.type == "adam") {
        adam_t_ += 1;
    }
    const float step_t = static_cast<float>(std::max<uint64_t>(1, adam_t_));

    size_t flat_idx = 0;
    auto step_tensor = [&](Tensor& param, bool apply_weight_decay) {
        for (size_t i = 0; i < param.data.size(); ++i, ++flat_idx) {
            float g = param.grad[i];
            if (apply_weight_decay && optimizer_config_.weight_decay > 0.0f) {
                g += optimizer_config_.weight_decay * param.data[i];
            }

            if (optimizer_config_.type == "adam") {
                const float b1 = optimizer_config_.beta1;
                const float b2 = optimizer_config_.beta2;
                const float eps = optimizer_config_.epsilon;
                adam_m_[flat_idx] = b1 * adam_m_[flat_idx] + (1.0f - b1) * g;
                adam_v_[flat_idx] = b2 * adam_v_[flat_idx] + (1.0f - b2) * g * g;
                const float mhat = adam_m_[flat_idx] / (1.0f - std::pow(b1, step_t));
                const float vhat = adam_v_[flat_idx] / (1.0f - std::pow(b2, step_t));
                param.data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
            } else {
                param.data[i] -= lr * g;
            }
        }
    };

    for (Linear& layer : layers_) {
        step_tensor(layer.weight, true);
        step_tensor(layer.bias, false);
    }
    for (size_t l = 0; l < gamma_.size(); ++l) {
        step_tensor(gamma_[l], false);
        step_tensor(beta_[l], false);
    }
}

void Model::zero_grad() {
    if (inference_only_) {
        return;
    }
    for (Linear& layer : layers_) {
        layer.weight.zero_grad();
        layer.bias.zero_grad();
    }
    for (size_t l = 0; l < gamma_.size(); ++l) {
        gamma_[l].zero_grad();
        beta_[l].zero_grad();
    }
}

void Model::set_training(bool is_training) {
    if (inference_only_) {
        training_ = false;
        return;
    }
    training_ = is_training;
}

void Model::set_inference_only(bool enabled) {
    inference_only_ = enabled;
    if (enabled) {
        training_ = false;
        use_dropout_ = false;
        layer_inputs_.clear();
        norm_outputs_.clear();
        dropout_masks_.clear();
        norm_xhat_cache_.clear();
        norm_var_cache_.clear();

        for (Linear& layer : layers_) {
            layer.weight.disable_grad_storage();
            layer.bias.disable_grad_storage();
        }
        for (size_t i = 0; i < gamma_.size(); ++i) {
            gamma_[i].disable_grad_storage();
            beta_[i].disable_grad_storage();
        }

        adam_m_.clear();
        adam_v_.clear();
        adam_m_.shrink_to_fit();
        adam_v_.shrink_to_fit();
    }
}

bool Model::is_inference_only() const {
    return inference_only_;
}

void Model::save(const std::string& path, size_t input_dim, size_t output_dim) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open model file for writing");
    }

    const size_t layer_count = layers_.size();
    out.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
    out.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
    out.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

    for (const Linear& layer : layers_) {
        out.write(reinterpret_cast<const char*>(&layer.in_dim), sizeof(layer.in_dim));
        out.write(reinterpret_cast<const char*>(&layer.out_dim), sizeof(layer.out_dim));

        const size_t w_size = layer.weight.data.size();
        out.write(reinterpret_cast<const char*>(&w_size), sizeof(w_size));
        out.write(reinterpret_cast<const char*>(layer.weight.data.data()), static_cast<std::streamsize>(w_size * sizeof(float)));

        const size_t b_size = layer.bias.data.size();
        out.write(reinterpret_cast<const char*>(&b_size), sizeof(b_size));
        out.write(reinterpret_cast<const char*>(layer.bias.data.data()), static_cast<std::streamsize>(b_size * sizeof(float)));
    }
}

void Model::save_state(const std::string& path,
                       size_t input_dim,
                       size_t output_dim,
                       const ModelConfig& model_config,
                       const ModelTrainingMetadata& metadata) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open model state file for writing");
    }

    const uint64_t arch_hash = architecture_hash(input_dim, output_dim, model_config);
    write_binary(out, kModelMagic);
    write_binary(out, kModelStateVersion);
    write_binary(out, arch_hash);

    write_binary(out, metadata.timestamp_unix);
    write_binary(out, metadata.epoch);
    write_binary(out, metadata.val_loss);
    write_binary(out, metadata.val_top1);
    write_string(out, metadata.optimizer);

    write_binary(out, input_dim);
    write_binary(out, output_dim);

    const size_t hidden_count = model_config.hidden_layers.size();
    write_binary(out, hidden_count);
    for (size_t h : model_config.hidden_layers) {
        write_binary(out, h);
    }

    write_string(out, model_config.activation);
    write_string(out, model_config.norm);
    write_binary(out, model_config.use_dropout);
    write_binary(out, model_config.dropout_prob);
    write_binary(out, model_config.leaky_relu_alpha);

    write_string(out, optimizer_config_.type);
    write_binary(out, optimizer_config_.beta1);
    write_binary(out, optimizer_config_.beta2);
    write_binary(out, optimizer_config_.epsilon);
    write_binary(out, optimizer_config_.weight_decay);
    write_binary(out, adam_t_);

    const size_t layer_count = layers_.size();
    write_binary(out, layer_count);
    for (const Linear& layer : layers_) {
        write_binary(out, layer.in_dim);
        write_binary(out, layer.out_dim);

        const size_t w_size = layer.weight.data.size();
        write_binary(out, w_size);
        out.write(reinterpret_cast<const char*>(layer.weight.data.data()), static_cast<std::streamsize>(w_size * sizeof(float)));

        const size_t b_size = layer.bias.data.size();
        write_binary(out, b_size);
        out.write(reinterpret_cast<const char*>(layer.bias.data.data()), static_cast<std::streamsize>(b_size * sizeof(float)));
    }

    const size_t norm_param_count = gamma_.size();
    write_binary(out, norm_param_count);
    for (size_t i = 0; i < norm_param_count; ++i) {
        const size_t g_size = gamma_[i].data.size();
        write_binary(out, g_size);
        out.write(reinterpret_cast<const char*>(gamma_[i].data.data()), static_cast<std::streamsize>(g_size * sizeof(float)));

        const size_t b_size = beta_[i].data.size();
        write_binary(out, b_size);
        out.write(reinterpret_cast<const char*>(beta_[i].data.data()), static_cast<std::streamsize>(b_size * sizeof(float)));
    }

    const size_t m_size = adam_m_.size();
    write_binary(out, m_size);
    if (m_size > 0) {
        out.write(reinterpret_cast<const char*>(adam_m_.data()), static_cast<std::streamsize>(m_size * sizeof(float)));
    }

    const size_t v_size = adam_v_.size();
    write_binary(out, v_size);
    if (v_size > 0) {
        out.write(reinterpret_cast<const char*>(adam_v_.data()), static_cast<std::streamsize>(v_size * sizeof(float)));
    }
}

bool Model::load_state(const std::string& path,
                       size_t input_dim,
                       size_t output_dim,
                       ModelFileInfo* info) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t file_arch_hash = 0;
    if (!read_binary(in, &magic) || !read_binary(in, &version) || !read_binary(in, &file_arch_hash)) {
        return false;
    }
    if (magic != kModelMagic || version != kModelStateVersion) {
        return false;
    }

    ModelTrainingMetadata metadata;
    if (!read_binary(in, &metadata.timestamp_unix) ||
        !read_binary(in, &metadata.epoch) ||
        !read_binary(in, &metadata.val_loss) ||
        !read_binary(in, &metadata.val_top1)) {
        return false;
    }
    metadata.optimizer = read_string(in);
    if (!in) {
        return false;
    }

    size_t ckpt_input = 0;
    size_t ckpt_output = 0;
    if (!read_binary(in, &ckpt_input) || !read_binary(in, &ckpt_output)) {
        return false;
    }
    if (ckpt_input != input_dim || ckpt_output != output_dim) {
        return false;
    }

    size_t hidden_count = 0;
    if (!read_binary(in, &hidden_count)) {
        return false;
    }
    std::vector<size_t> hidden_layers(hidden_count, 0);
    for (size_t i = 0; i < hidden_count; ++i) {
        if (!read_binary(in, &hidden_layers[i])) {
            return false;
        }
    }

    ModelConfig file_config;
    file_config.hidden_layers = hidden_layers;
    file_config.activation = read_string(in);
    file_config.norm = read_string(in);
    if (!read_binary(in, &file_config.use_dropout) ||
        !read_binary(in, &file_config.dropout_prob) ||
        !read_binary(in, &file_config.leaky_relu_alpha)) {
        return false;
    }

    const uint64_t expected_hash = architecture_hash(input_dim, output_dim, file_config);
    if (file_arch_hash != expected_hash) {
        return false;
    }

    optimizer_config_.type = read_string(in);
    if (!read_binary(in, &optimizer_config_.beta1) ||
        !read_binary(in, &optimizer_config_.beta2) ||
        !read_binary(in, &optimizer_config_.epsilon) ||
        !read_binary(in, &optimizer_config_.weight_decay) ||
        !read_binary(in, &adam_t_)) {
        return false;
    }

    size_t layer_count = 0;
    if (!read_binary(in, &layer_count) || layer_count != layers_.size()) {
        return false;
    }

    for (size_t i = 0; i < layer_count; ++i) {
        size_t in_dim = 0;
        size_t out_dim = 0;
        if (!read_binary(in, &in_dim) || !read_binary(in, &out_dim)) {
            return false;
        }
        if (in_dim != layers_[i].in_dim || out_dim != layers_[i].out_dim) {
            return false;
        }

        size_t w_size = 0;
        if (!read_binary(in, &w_size) || w_size != layers_[i].weight.data.size()) {
            return false;
        }
        in.read(reinterpret_cast<char*>(layers_[i].weight.data.data()), static_cast<std::streamsize>(w_size * sizeof(float)));
        if (!in) {
            return false;
        }

        size_t b_size = 0;
        if (!read_binary(in, &b_size) || b_size != layers_[i].bias.data.size()) {
            return false;
        }
        in.read(reinterpret_cast<char*>(layers_[i].bias.data.data()), static_cast<std::streamsize>(b_size * sizeof(float)));
        if (!in) {
            return false;
        }
    }

    size_t norm_param_count = 0;
    if (!read_binary(in, &norm_param_count) || norm_param_count != gamma_.size()) {
        return false;
    }
    for (size_t i = 0; i < norm_param_count; ++i) {
        size_t g_size = 0;
        if (!read_binary(in, &g_size) || g_size != gamma_[i].data.size()) {
            return false;
        }
        in.read(reinterpret_cast<char*>(gamma_[i].data.data()), static_cast<std::streamsize>(g_size * sizeof(float)));
        if (!in) {
            return false;
        }

        size_t b_size = 0;
        if (!read_binary(in, &b_size) || b_size != beta_[i].data.size()) {
            return false;
        }
        in.read(reinterpret_cast<char*>(beta_[i].data.data()), static_cast<std::streamsize>(b_size * sizeof(float)));
        if (!in) {
            return false;
        }
    }

    size_t m_size = 0;
    if (!read_binary(in, &m_size)) {
        return false;
    }
    adam_m_.assign(m_size, 0.0f);
    if (m_size > 0) {
        in.read(reinterpret_cast<char*>(adam_m_.data()), static_cast<std::streamsize>(m_size * sizeof(float)));
        if (!in) {
            return false;
        }
    }

    size_t v_size = 0;
    if (!read_binary(in, &v_size)) {
        return false;
    }
    adam_v_.assign(v_size, 0.0f);
    if (v_size > 0) {
        in.read(reinterpret_cast<char*>(adam_v_.data()), static_cast<std::streamsize>(v_size * sizeof(float)));
        if (!in) {
            return false;
        }
    }

    if (info != nullptr) {
        info->version = version;
        info->architecture_hash = file_arch_hash;
        info->metadata = metadata;
    }
    return true;
}

void Model::copy_parameters_from(const Model& other) {
    if (layers_.size() != other.layers_.size() || gamma_.size() != other.gamma_.size()) {
        throw std::runtime_error("copy_parameters_from: incompatible model layouts");
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].weight.data.size() != other.layers_[i].weight.data.size() ||
            layers_[i].bias.data.size() != other.layers_[i].bias.data.size()) {
            throw std::runtime_error("copy_parameters_from: incompatible layer parameter sizes");
        }
        layers_[i].weight.data = other.layers_[i].weight.data;
        layers_[i].bias.data = other.layers_[i].bias.data;
    }
    for (size_t i = 0; i < gamma_.size(); ++i) {
        gamma_[i].data = other.gamma_[i].data;
        beta_[i].data = other.beta_[i].data;
    }
}

size_t Model::gradient_size() const {
    size_t total = 0;
    for (const Linear& layer : layers_) {
        total += layer.weight.grad.size();
        total += layer.bias.grad.size();
    }
    for (size_t i = 0; i < gamma_.size(); ++i) {
        total += gamma_[i].grad.size();
        total += beta_[i].grad.size();
    }
    return total;
}

void Model::gradients_to_vector(std::vector<float>& out) const {
    out.clear();
    out.reserve(gradient_size());
    for (const Linear& layer : layers_) {
        out.insert(out.end(), layer.weight.grad.begin(), layer.weight.grad.end());
        out.insert(out.end(), layer.bias.grad.begin(), layer.bias.grad.end());
    }
    for (size_t i = 0; i < gamma_.size(); ++i) {
        out.insert(out.end(), gamma_[i].grad.begin(), gamma_[i].grad.end());
        out.insert(out.end(), beta_[i].grad.begin(), beta_[i].grad.end());
    }
}

void Model::set_gradients_from_vector(const std::vector<float>& values) {
    if (values.size() != gradient_size()) {
        throw std::runtime_error("set_gradients_from_vector: incorrect gradient vector size");
    }

    size_t idx = 0;
    auto fill_tensor_grad = [&](Tensor& tensor) {
        for (size_t i = 0; i < tensor.grad.size(); ++i, ++idx) {
            tensor.grad[i] = values[idx];
        }
    };

    for (Linear& layer : layers_) {
        fill_tensor_grad(layer.weight);
        fill_tensor_grad(layer.bias);
    }
    for (size_t i = 0; i < gamma_.size(); ++i) {
        fill_tensor_grad(gamma_[i]);
        fill_tensor_grad(beta_[i]);
    }
}

ModelDecision Model::decide(const WorldSnapshot& world_snapshot, uint16_t controlled_country_id) {
    ModelDecision decision;
    decision.actor_country_id = controlled_country_id;

    const CountrySnapshot* self = nullptr;
    for (const CountrySnapshot& country : world_snapshot.countries) {
        if (country.id == controlled_country_id) {
            self = &country;
            break;
        }
    }
    if (self == nullptr) {
        decision.strategy = Strategy::Defend;
        return decision;
    }

    auto total_strength = [](const CountrySnapshot& country) -> int64_t {
        return country.units_infantry_milli + (country.units_armor_milli * 3) / 2 +
               (country.units_artillery_milli * 5) / 4 + (country.units_air_fighter_milli * 7) / 5 +
               (country.units_air_bomber_milli * 8) / 5 + (country.units_naval_surface_milli * 7) / 5 +
               (country.units_naval_submarine_milli * 3) / 2;
    };

    auto norm_percent_milli = [](int64_t milli) -> float {
        return std::clamp(static_cast<float>(milli) / 100000.0f, 0.0f, 1.0f);
    };
    auto norm_unit_milli = [](int64_t milli) -> float {
        return std::clamp(static_cast<float>(milli) / 1000.0f, 0.0f, 1.0f);
    };
    auto norm_signed_percent_milli = [](int64_t milli) -> float {
        return std::clamp(0.5f + static_cast<float>(milli) / 200000.0f, 0.0f, 1.0f);
    };

    int64_t self_strength = std::max<int64_t>(1, total_strength(*self));
    int64_t strongest_neighbor_strength = 0;
    uint16_t strongest_neighbor_id = 0;
    int64_t weakest_neighbor_strength = std::numeric_limits<int64_t>::max();
    uint16_t weakest_neighbor_id = 0;
    int64_t total_neighbor_strength = 0;
    int64_t strongest_believed_neighbor_strength = 0;
    uint16_t strongest_believed_neighbor_id = 0;
    int64_t weakest_believed_neighbor_strength = std::numeric_limits<int64_t>::max();
    uint16_t weakest_believed_neighbor_id = 0;
    int64_t total_believed_neighbor_strength = 0;
    int64_t max_neighbor_supply_gap = 0;
    uint16_t best_trade_target_id = 0;
    float best_trade_target_score = -1e9f;
    uint16_t best_treaty_target_id = 0;
    float best_treaty_target_score = -1e9f;
    uint16_t embargo_target_id = 0;
    float embargo_target_score = -1e9f;
    float best_enemy_intel = 0.0f;
    float worst_enemy_intel = 1.0f;
    float total_enemy_intel = 0.0f;
    size_t intel_count = 0;
    bool strongest_target_second_strike = false;
    float strongest_target_escalation = 0.0f;
    float strongest_target_supply_weakness = 0.0f;
    float strongest_target_treaty_entanglement = 0.0f;
    float highest_neighbor_trust = 0.0f;
    float lowest_neighbor_trust = 1.0f;
    float total_neighbor_trust = 0.0f;
    size_t neighbor_trust_count = 0;
    float max_opponent_model_confidence = 0.0f;
    float total_opponent_model_confidence = 0.0f;
    size_t opponent_model_count = 0;

    for (const CountrySnapshot& other : world_snapshot.countries) {
        if (other.id == controlled_country_id) {
            continue;
        }

        const bool adjacent = std::find(self->adjacent_country_ids.begin(), self->adjacent_country_ids.end(), other.id) !=
            self->adjacent_country_ids.end();
        const int64_t s = total_strength(other);
        const auto believed_it = self->believed_army_size_milli.find(other.id);
        const int64_t believed_strength = believed_it == self->believed_army_size_milli.end() ? s : std::max<int64_t>(1, believed_it->second);
        const auto trust_it = self->trust_in_milli.find(other.id);
        const float bilateral_trust = trust_it == self->trust_in_milli.end()
            ? norm_percent_milli(other.trust_average_milli)
            : norm_percent_milli(trust_it->second);
        const auto confidence_it = self->opponent_model_confidence_milli.find(other.id);
        const float model_confidence = confidence_it == self->opponent_model_confidence_milli.end()
            ? norm_percent_milli(self->intelligence_milli)
            : norm_percent_milli(confidence_it->second);
        if (adjacent) {
            total_neighbor_strength += s;
            if (s > strongest_neighbor_strength) {
                strongest_neighbor_strength = s;
                strongest_neighbor_id = other.id;
                strongest_target_second_strike = other.second_strike_capable;
                strongest_target_escalation = norm_percent_milli(other.escalation_level_milli * 20000);
                strongest_target_supply_weakness = 1.0f - norm_percent_milli(other.supply_level_milli);
                strongest_target_treaty_entanglement = std::clamp(static_cast<float>(other.defense_pact_ids.size() + other.non_aggression_pact_ids.size()) / 10.0f, 0.0f, 1.0f);
            }
            if (s < weakest_neighbor_strength) {
                weakest_neighbor_strength = s;
                weakest_neighbor_id = other.id;
            }
            total_believed_neighbor_strength += believed_strength;
            if (believed_strength > strongest_believed_neighbor_strength) {
                strongest_believed_neighbor_strength = believed_strength;
                strongest_believed_neighbor_id = other.id;
            }
            if (believed_strength < weakest_believed_neighbor_strength) {
                weakest_believed_neighbor_strength = believed_strength;
                weakest_believed_neighbor_id = other.id;
            }
            const auto intel_it = self->intel_on_enemy_milli.find(other.id);
            const float intel_score = intel_it == self->intel_on_enemy_milli.end() ? norm_percent_milli(self->intelligence_milli) : norm_percent_milli(intel_it->second);
            best_enemy_intel = std::max(best_enemy_intel, intel_score);
            worst_enemy_intel = std::min(worst_enemy_intel, intel_score);
            total_enemy_intel += intel_score;
            ++intel_count;
            max_neighbor_supply_gap = std::max<int64_t>(max_neighbor_supply_gap, other.supply_capacity_milli - other.supply_level_milli);
            highest_neighbor_trust = std::max(highest_neighbor_trust, bilateral_trust);
            lowest_neighbor_trust = std::min(lowest_neighbor_trust, bilateral_trust);
            total_neighbor_trust += bilateral_trust;
            ++neighbor_trust_count;
            max_opponent_model_confidence = std::max(max_opponent_model_confidence, model_confidence);
            total_opponent_model_confidence += model_confidence;
            ++opponent_model_count;
        }

        const float trust_gap = bilateral_trust;
        const float complement = (1.0f - norm_percent_milli(other.trade_balance_milli)) + norm_percent_milli(other.industry_milli);
        const float partner_score = trust_gap * 1.8f + complement * 0.9f - norm_percent_milli(other.coup_risk_milli) * 0.8f;
        if (partner_score > best_trade_target_score) {
            best_trade_target_score = partner_score;
            best_trade_target_id = other.id;
        }

        const float treaty_score = norm_percent_milli(other.reputation_milli) * 1.2f + trust_gap * 1.4f +
            (1.0f - norm_percent_milli(other.escalation_level_milli * 20000)) * 0.9f -
            norm_percent_milli(other.coup_risk_milli) * 0.6f;
        if (treaty_score > best_treaty_target_score) {
            best_treaty_target_score = treaty_score;
            best_treaty_target_id = other.id;
        }

        const float embargo_score = norm_percent_milli(other.trade_balance_milli) * 1.1f + norm_percent_milli(other.industry_milli) * 0.9f +
            norm_percent_milli(other.faction_industrial_milli) * 0.7f - trust_gap * 0.6f;
        if (embargo_score > embargo_target_score) {
            embargo_target_score = embargo_score;
            embargo_target_id = other.id;
        }
    }
    if (weakest_neighbor_strength == std::numeric_limits<int64_t>::max()) {
        weakest_neighbor_strength = 0;
    }
    if (weakest_believed_neighbor_strength == std::numeric_limits<int64_t>::max()) {
        weakest_believed_neighbor_strength = 0;
    }

    const size_t in_dim = input_dim();
    const size_t out_dim = output_dim();
    if (in_dim != battle_common::kBattleInputDim || out_dim != battle_common::kBattleOutputDim) {
        throw std::runtime_error(
            "Model::decide requires battle model dimensions " +
            std::to_string(battle_common::kBattleInputDim) + "x" + std::to_string(battle_common::kBattleOutputDim) +
            ", got " + std::to_string(in_dim) + "x" + std::to_string(out_dim));
    }

    std::array<float, battle_common::kBattleBaseInputDim> base_frame{};
    auto set_feature = [&](size_t idx, float value) {
        if (idx < base_frame.size()) {
            base_frame[idx] = value;
        }
    };

    const float self_strength_norm = static_cast<float>(std::min<int64_t>(self_strength, 1'000'000)) / 1'000'000.0f;
    const float neighbor_strength_norm = static_cast<float>(std::min<int64_t>(strongest_neighbor_strength, 1'000'000)) / 1'000'000.0f;
    const float strength_delta_norm = std::clamp(0.5f + (self_strength_norm - neighbor_strength_norm) * 0.8f, 0.0f, 1.0f);
    const float weakest_neighbor_norm = static_cast<float>(std::max<int64_t>(0, weakest_neighbor_strength)) / 1'000'000.0f;
    const float avg_neighbor_norm = static_cast<float>(std::min<int64_t>(1'000'000, total_neighbor_strength / std::max<int64_t>(1, static_cast<int64_t>(std::max<size_t>(1, self->adjacent_country_ids.size()))))) / 1'000'000.0f;
    const float strongest_believed_neighbor_norm = static_cast<float>(std::min<int64_t>(strongest_believed_neighbor_strength, 1'000'000)) / 1'000'000.0f;
    const float weakest_believed_neighbor_norm = static_cast<float>(std::max<int64_t>(0, weakest_believed_neighbor_strength)) / 1'000'000.0f;
    const float avg_believed_neighbor_norm = static_cast<float>(std::min<int64_t>(1'000'000, total_believed_neighbor_strength / std::max<int64_t>(1, static_cast<int64_t>(std::max<size_t>(1, self->adjacent_country_ids.size()))))) / 1'000'000.0f;
    const float morale_norm = norm_percent_milli(self->civilian_morale_milli);
    const float econ_norm = norm_percent_milli(self->economic_stability_milli);
    const float logistics_norm = norm_percent_milli(self->logistics_milli);
    const float intelligence_norm = norm_percent_milli(self->intelligence_milli);
    const float industry_norm = norm_percent_milli(self->industry_milli);
    const float tech_norm = norm_percent_milli(self->technology_milli);
    const float reserve_norm = norm_percent_milli(self->resource_reserve_milli);
    const float supply_level_norm = norm_percent_milli(self->supply_level_milli);
    const float supply_capacity_norm = norm_percent_milli(self->supply_capacity_milli);
    const float reputation_norm = norm_percent_milli(self->reputation_milli);
    const float escalation_norm = std::clamp(static_cast<float>(self->escalation_level_milli) / 5000.0f, 0.0f, 1.0f);
    const float second_strike_norm = self->second_strike_capable ? 1.0f : 0.0f;
    const float stance_norm = static_cast<float>(self->diplomatic_stance) / 2.0f;
    const float has_target = strongest_neighbor_id == 0 ? 0.0f : 1.0f;
    const float tick_norm = static_cast<float>(world_snapshot.tick % 1000) / 1000.0f;
    const float weather_norm = norm_percent_milli(self->weather_severity_milli);
    const float season_norm = norm_percent_milli(self->seasonal_effect_milli);
    const float supply_norm = norm_percent_milli(self->supply_stockpile_milli);
    const float mountain_norm = norm_unit_milli(self->terrain_mountains_milli);
    const float forest_norm = norm_unit_milli(self->terrain_forests_milli);
    const float urban_norm = norm_unit_milli(self->terrain_urban_milli);
    const float missile_defense_norm = norm_percent_milli(self->tech_missile_defense_milli);
    const float cyber_norm = norm_percent_milli(self->tech_cyber_warfare_milli);
    const float ew_norm = norm_percent_milli(self->tech_electronic_warfare_milli);
    const float drone_norm = norm_percent_milli(self->tech_drone_ops_milli);
    const float oil_norm = norm_percent_milli(self->resource_oil_milli);
    const float mineral_norm = norm_percent_milli(self->resource_minerals_milli);
    const float food_norm = norm_percent_milli(self->resource_food_milli);
    const float rare_earth_norm = norm_percent_milli(self->resource_rare_earth_milli);
    const float gov_norm = norm_percent_milli(self->gov_stability_milli);
    const float dissent_norm = norm_percent_milli(self->public_dissent_milli);
    const float corruption_norm = norm_percent_milli(self->corruption_milli);
    const float nuclear_norm = norm_percent_milli(self->nuclear_readiness_milli);
    const float deterrence_norm = norm_percent_milli(self->deterrence_posture_milli);
    const float trust_norm = norm_percent_milli(self->trust_average_milli);
    const float trade_balance_norm = norm_signed_percent_milli(self->trade_balance_milli);
    const float trade_partner_ratio = std::clamp(static_cast<float>(self->trade_partner_ids.size()) / 8.0f, 0.0f, 1.0f);
    const float defense_pact_ratio = std::clamp(static_cast<float>(self->defense_pact_ids.size()) / 6.0f, 0.0f, 1.0f);
    const float non_aggression_ratio = std::clamp(static_cast<float>(self->non_aggression_pact_ids.size()) / 6.0f, 0.0f, 1.0f);
    const float trade_treaty_ratio = std::clamp(static_cast<float>(self->trade_treaty_ids.size()) / 6.0f, 0.0f, 1.0f);
    const float oil_reserve_norm = norm_percent_milli(self->resource_oil_reserves_milli);
    const float mineral_reserve_norm = norm_percent_milli(self->resource_minerals_reserves_milli);
    const float food_reserve_norm = norm_percent_milli(self->resource_food_reserves_milli);
    const float rare_reserve_norm = norm_percent_milli(self->resource_rare_earth_reserves_milli);
    const float upkeep_norm = norm_percent_milli(self->military_upkeep_milli);
    const float faction_military_norm = norm_percent_milli(self->faction_military_milli);
    const float faction_industrial_norm = norm_percent_milli(self->faction_industrial_milli);
    const float faction_civilian_norm = norm_percent_milli(self->faction_civilian_milli);
    const float coup_risk_norm = norm_percent_milli(self->coup_risk_milli);
    const float election_urgency_norm = std::clamp(1.0f - static_cast<float>(std::max(0, self->election_cycle)) / 20.0f, 0.0f, 1.0f);
    const float draft_norm = norm_percent_milli(self->draft_level_milli);
    const float war_weariness_norm = norm_percent_milli(self->war_weariness_milli);
    const float adjacent_ratio = std::clamp(static_cast<float>(self->adjacent_country_ids.size()) / 12.0f, 0.0f, 1.0f);
    const float ally_ratio = self->allied_country_ids.empty()
        ? 0.0f
        : std::clamp(static_cast<float>(self->allied_country_ids.size()) / 8.0f, 0.0f, 1.0f);
    const float threat_ratio = self_strength_norm > 1e-5f
        ? std::clamp(neighbor_strength_norm / self_strength_norm, 0.0f, 2.0f)
        : 1.0f;
    const float naval_ratio = self_strength > 0
        ? std::clamp(static_cast<float>(self->units_naval_surface_milli + self->units_naval_submarine_milli) / static_cast<float>(self_strength), 0.0f, 1.0f)
        : 0.0f;
    const float air_ratio = self_strength > 0
        ? std::clamp(static_cast<float>(self->units_air_fighter_milli + self->units_air_bomber_milli) / static_cast<float>(self_strength), 0.0f, 1.0f)
        : 0.0f;
    const float cycle = 0.5f + 0.5f * std::sin(static_cast<float>(world_snapshot.tick) * 0.07f);
    const float reserve_stress = std::clamp(1.0f - (oil_reserve_norm + mineral_reserve_norm + food_reserve_norm + rare_reserve_norm) / 4.0f + upkeep_norm * 0.3f,
                                            0.0f,
                                            1.0f);
    const float neighbor_trust_avg = neighbor_trust_count == 0 ? trust_norm : total_neighbor_trust / static_cast<float>(neighbor_trust_count);
    const float neighbor_trust_low = neighbor_trust_count == 0 ? trust_norm : lowest_neighbor_trust;
    const float neighbor_trust_high = neighbor_trust_count == 0 ? trust_norm : highest_neighbor_trust;
    const float opponent_confidence_avg = opponent_model_count == 0 ? intelligence_norm : total_opponent_model_confidence / static_cast<float>(opponent_model_count);
    const float avg_enemy_intel = intel_count == 0 ? intelligence_norm : total_enemy_intel / static_cast<float>(intel_count);
    const float recent_betrayals_norm = std::clamp(static_cast<float>(std::max(0, self->recent_betrayals)) / 6.0f, 0.0f, 1.0f);
    const float strategic_depth_norm = std::clamp(static_cast<float>(std::max<int64_t>(0, self->strategic_depth_milli)) / 6000.0f, 0.0f, 1.0f);
    const float attack_opportunity = std::clamp((strength_delta_norm * 0.38f + best_enemy_intel * 0.16f + supply_capacity_norm * 0.12f +
                                                 (1.0f - strongest_target_supply_weakness) * 0.08f + opponent_confidence_avg * 0.10f +
                                                 strategic_depth_norm * 0.12f) - non_aggression_ratio * 0.15f - recent_betrayals_norm * 0.06f,
                                                0.0f,
                                                1.0f);
    const float crisis_pressure = std::clamp((1.0f - morale_norm) * 0.22f + (1.0f - econ_norm) * 0.22f + coup_risk_norm * 0.18f +
                                             war_weariness_norm * 0.15f + escalation_norm * 0.13f + recent_betrayals_norm * 0.10f,
                                            0.0f,
                                            1.0f);
    const float intel_opportunity = std::clamp((1.0f - avg_enemy_intel) * 0.60f + strongest_target_supply_weakness * 0.20f + has_target * 0.20f, 0.0f, 1.0f);

    set_feature(0, norm_percent_milli(self->units_infantry_milli));
    set_feature(1, norm_percent_milli(self->units_armor_milli * 2));
    set_feature(2, norm_percent_milli(self->units_artillery_milli * 2));
    set_feature(3, norm_percent_milli(self->units_air_fighter_milli * 4));
    set_feature(4, norm_percent_milli(self->units_air_bomber_milli * 4));
    set_feature(5, norm_percent_milli(self->units_naval_surface_milli * 5));
    set_feature(6, norm_percent_milli(self->units_naval_submarine_milli * 6));
    set_feature(7, supply_level_norm);
    set_feature(8, supply_capacity_norm);
    set_feature(9, econ_norm);
    set_feature(10, morale_norm);
    set_feature(11, logistics_norm);
    set_feature(12, intelligence_norm);
    set_feature(13, industry_norm);
    set_feature(14, tech_norm);
    set_feature(15, reserve_norm);
    set_feature(16, reputation_norm);
    set_feature(17, escalation_norm);
    set_feature(18, second_strike_norm);
    set_feature(19, stance_norm);
    set_feature(20, weather_norm);
    set_feature(21, season_norm);
    set_feature(22, supply_norm);
    set_feature(23, mountain_norm);
    set_feature(24, forest_norm);
    set_feature(25, urban_norm);
    set_feature(26, missile_defense_norm);
    set_feature(27, cyber_norm);
    set_feature(28, ew_norm);
    set_feature(29, drone_norm);
    set_feature(30, oil_norm);
    set_feature(31, mineral_norm);
    set_feature(32, food_norm);
    set_feature(33, rare_earth_norm);
    set_feature(34, oil_reserve_norm);
    set_feature(35, mineral_reserve_norm);
    set_feature(36, food_reserve_norm);
    set_feature(37, rare_reserve_norm);
    set_feature(38, upkeep_norm);
    set_feature(39, faction_military_norm);
    set_feature(40, faction_industrial_norm);
    set_feature(41, faction_civilian_norm);
    set_feature(42, gov_norm);
    set_feature(43, dissent_norm);
    set_feature(44, corruption_norm);
    set_feature(45, coup_risk_norm);
    set_feature(46, draft_norm);
    set_feature(47, war_weariness_norm);
    set_feature(48, trade_balance_norm);
    set_feature(49, trade_partner_ratio);
    set_feature(50, defense_pact_ratio);
    set_feature(51, non_aggression_ratio);
    set_feature(52, trade_treaty_ratio);
    set_feature(53, adjacent_ratio);
    set_feature(54, ally_ratio);
    set_feature(55, neighbor_strength_norm);
    set_feature(56, weakest_neighbor_norm);
    set_feature(57, avg_neighbor_norm);
    set_feature(58, strength_delta_norm);
    set_feature(59, best_enemy_intel);
    set_feature(60, intel_count == 0 ? 0.0f : worst_enemy_intel);
    set_feature(61, avg_enemy_intel);
    set_feature(62, std::clamp(best_trade_target_score / 3.0f, 0.0f, 1.0f));
    set_feature(63, std::clamp(best_treaty_target_score / 3.0f, 0.0f, 1.0f));
    set_feature(64, std::clamp(embargo_target_score / 3.0f, 0.0f, 1.0f));
    set_feature(65, attack_opportunity);
    set_feature(66, crisis_pressure);
    set_feature(67, nuclear_norm);
    set_feature(68, deterrence_norm);
    set_feature(69, trust_norm);
    set_feature(70, cycle);
    set_feature(71, has_target);
    set_feature(72, strongest_target_second_strike ? 1.0f : 0.0f);
    set_feature(73, strongest_target_escalation);
    set_feature(74, strongest_target_supply_weakness);
    set_feature(75, strongest_target_treaty_entanglement);
    set_feature(76, naval_ratio);
    set_feature(77, air_ratio);
    set_feature(78, reserve_stress);
    set_feature(79, std::clamp(static_cast<float>(max_neighbor_supply_gap) / 100000.0f, 0.0f, 1.0f));
    set_feature(80, strongest_believed_neighbor_norm);
    set_feature(81, weakest_believed_neighbor_norm);
    set_feature(82, avg_believed_neighbor_norm);
    set_feature(83, neighbor_trust_low);
    set_feature(84, neighbor_trust_high);
    set_feature(85, neighbor_trust_avg);
    set_feature(86, opponent_confidence_avg);
    set_feature(87, max_opponent_model_confidence);
    set_feature(88, recent_betrayals_norm);
    set_feature(89, strategic_depth_norm);

    auto& history = temporal_history_[controlled_country_id];
    history.push_back(base_frame);
    while (history.size() > battle_common::kBattleTemporalWindow) {
        history.pop_front();
    }

    Tensor nn_input({1, in_dim}, 0.0f);
    for (size_t t = 0; t < battle_common::kBattleTemporalWindow; ++t) {
        const size_t offset = t * battle_common::kBattleBaseInputDim;
        if (offset + battle_common::kBattleBaseInputDim > nn_input.data.size()) {
            break;
        }
        const bool has_frame = t < history.size();
        if (!has_frame) {
            continue;
        }
        const size_t source_idx = history.size() - 1U - t;
        const auto& frame = history[source_idx];
        for (size_t f = 0; f < battle_common::kBattleBaseInputDim; ++f) {
            nn_input.data[offset + f] = frame[f];
        }
    }

    Tensor logits = forward(nn_input);
    const size_t action_count = std::min<size_t>(battle_common::kBattlePolicyActionDim, logits.data.size());
    if (action_count == 0) {
        decision.strategy = Strategy::Defend;
        decision.target_country_id = strongest_neighbor_id;
        return decision;
    }

    auto head_argmax = [&](size_t offset, size_t size) -> size_t {
        if (offset >= logits.data.size() || size == 0) {
            return 0;
        }
        const size_t capped = std::min(size, logits.data.size() - offset);
        size_t best = 0;
        float best_score = logits.data[offset];
        for (size_t i = 1; i < capped; ++i) {
            if (logits.data[offset + i] > best_score) {
                best_score = logits.data[offset + i];
                best = i;
            }
        }
        return best;
    };

    std::array<float, battle_common::kBattleStrategicGoalDim> strategic_logits{};
    for (size_t i = 0; i < strategic_logits.size(); ++i) {
        const size_t idx = battle_common::kBattleHeadStrategicOffset + i;
        strategic_logits[i] = idx < logits.data.size() ? logits.data[idx] : -1e9f;
    }

    float max_strategic = strategic_logits[0];
    for (size_t i = 1; i < strategic_logits.size(); ++i) {
        max_strategic = std::max(max_strategic, strategic_logits[i]);
    }
    std::array<float, battle_common::kBattleStrategicGoalDim> strategic_prob{};
    float strategic_sum = 0.0f;
    for (size_t i = 0; i < strategic_prob.size(); ++i) {
        strategic_prob[i] = std::exp(strategic_logits[i] - max_strategic);
        strategic_sum += strategic_prob[i];
    }
    if (strategic_sum > 0.0f) {
        for (float& p : strategic_prob) {
            p /= strategic_sum;
        }
    }

    enum class StrategicGoal : uint8_t { Attack = 0, Defend = 1, Trade = 2, Develop = 3 };

    auto action_belongs_to_goal = [](Strategy strategy, StrategicGoal goal) {
        switch (goal) {
            case StrategicGoal::Attack:
                return strategy == Strategy::Attack || strategy == Strategy::DeployUnits ||
                       strategy == Strategy::CyberAttack || strategy == Strategy::CyberOperation ||
                       strategy == Strategy::TacticalNuke || strategy == Strategy::StrategicNuke ||
                       strategy == Strategy::Betray || strategy == Strategy::BreakTreaty;
            case StrategicGoal::Defend:
                return strategy == Strategy::Defend || strategy == Strategy::RequestIntel ||
                       strategy == Strategy::ProposeDefensePact || strategy == Strategy::ProposeNonAggression ||
                       strategy == Strategy::SuppressDissent || strategy == Strategy::TransferWeapons;
            case StrategicGoal::Trade:
                return strategy == Strategy::SignTradeAgreement || strategy == Strategy::CancelTradeAgreement ||
                       strategy == Strategy::ImposeEmbargo || strategy == Strategy::InvestInResourceExtraction ||
                       strategy == Strategy::ReduceMilitaryUpkeep;
            case StrategicGoal::Develop:
                return strategy == Strategy::FocusEconomy || strategy == Strategy::DevelopTechnology ||
                       strategy == Strategy::FormAlliance || strategy == Strategy::Negotiate ||
                       strategy == Strategy::HoldElections || strategy == Strategy::CoupAttempt;
        }
        return false;
    };

    const size_t predicted_opponent_idx = head_argmax(battle_common::kBattleHeadOpponentOffset,
                                                      battle_common::kBattleOpponentActionDim);
    const Strategy predicted_opponent_action = static_cast<Strategy>(predicted_opponent_idx);
    const bool opponent_escalating =
        predicted_opponent_action == Strategy::Attack ||
        predicted_opponent_action == Strategy::DeployUnits ||
        predicted_opponent_action == Strategy::CyberAttack ||
        predicted_opponent_action == Strategy::TacticalNuke ||
        predicted_opponent_action == Strategy::StrategicNuke;

    const float value_logit = battle_common::kBattleHeadValueOffset < logits.data.size()
        ? logits.data[battle_common::kBattleHeadValueOffset]
        : 0.0f;
    const float value_estimate = std::tanh(value_logit);

    auto mcts_rollout = [&](StrategicGoal root_goal) {
        const size_t depth = 3;
        const float opp_penalty = opponent_escalating ? 0.10f : -0.04f;
        const float crisis_term = crisis_pressure * 0.12f;

        float best = -1e9f;
        for (size_t a = 0; a < strategic_prob.size(); ++a) {
            float node = 0.52f * strategic_prob[static_cast<size_t>(root_goal)] +
                         0.48f * strategic_prob[a];
            float future = node;
            for (size_t d = 0; d < depth; ++d) {
                const float blend = 0.60f - 0.10f * static_cast<float>(d);
                future = blend * future + (1.0f - blend) * strategic_prob[a];
            }
            const float score = future + value_estimate * 0.35f - crisis_term - opp_penalty;
            best = std::max(best, score);
        }
        return best;
    };

    StrategicGoal chosen_goal = StrategicGoal::Defend;
    float best_goal_score = -1e9f;
    for (size_t i = 0; i < strategic_prob.size(); ++i) {
        const StrategicGoal g = static_cast<StrategicGoal>(i);
        const float score = mcts_rollout(g);
        if (score > best_goal_score) {
            best_goal_score = score;
            chosen_goal = g;
        }
    }

    size_t best_idx = 0;
    float best_score = -1e9f;
    for (size_t i = 0; i < action_count; ++i) {
        const Strategy candidate = static_cast<Strategy>(i);
        if (!action_belongs_to_goal(candidate, chosen_goal)) {
            continue;
        }
        const float score = logits.data[i] + strategic_prob[static_cast<size_t>(chosen_goal)] * 0.30f;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_score < -1e8f) {
        best_idx = 0;
        best_score = logits.data[0];
        for (size_t i = 1; i < action_count; ++i) {
            if (logits.data[i] > best_score) {
                best_score = logits.data[i];
                best_idx = i;
            }
        }
    }

    const uint16_t primary_threat_id = strongest_believed_neighbor_id != 0 ? strongest_believed_neighbor_id : strongest_neighbor_id;
    const uint16_t opportunistic_target_id = weakest_believed_neighbor_id != 0 ? weakest_believed_neighbor_id : weakest_neighbor_id;

    decision.strategy = static_cast<Strategy>(best_idx);

    const size_t target_bucket = head_argmax(battle_common::kBattleHeadTargetBucketOffset,
                                             battle_common::kBattleTacticalTargetBucketDim);
    if (target_bucket == 0) {
        decision.target_country_id = primary_threat_id;
    } else if (target_bucket == 1) {
        decision.target_country_id = opportunistic_target_id;
    } else {
        decision.target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : best_trade_target_id;
    }

    const size_t commitment_bucket = head_argmax(battle_common::kBattleHeadCommitmentOffset,
                                                 battle_common::kBattleCommitmentBucketDim);
    if (commitment_bucket == 0) {
        decision.force_commitment = model_tuning::kCommitmentBucketLow;
    } else if (commitment_bucket == 1) {
        decision.force_commitment = model_tuning::kCommitmentBucketMedium;
    } else {
        decision.force_commitment = model_tuning::kCommitmentBucketHigh;
    }
    if (opponent_escalating) {
        decision.force_commitment = std::min(model_tuning::kCommitmentEscalationCap,
                                             decision.force_commitment + model_tuning::kCommitmentEscalationBoost);
    }

    const size_t allocation_bucket = head_argmax(battle_common::kBattleHeadAllocationOffset,
                                                 battle_common::kBattleAllocationBucketDim);
    if (allocation_bucket == 0) {
        decision.allocation = model_tuning::kAllocationBucketMilitary;
    } else if (allocation_bucket == 1) {
        decision.allocation = model_tuning::kAllocationBucketIndustry;
    } else {
        decision.allocation = model_tuning::kAllocationBucketCivilian;
    }

    auto populate_action = [&](Strategy strategy, uint16_t* target_country_id, NegotiationTerms* terms) {
        if (target_country_id == nullptr || terms == nullptr) {
            return;
        }
        if (strategy == Strategy::Attack) {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : embargo_target_id;
        } else if (strategy == Strategy::Negotiate) {
            *target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : primary_threat_id;
            if (self->diplomatic_stance == 2) {
                terms->type = "alliance";
                terms->details = "Mutual defense and non-aggression";
            } else {
                terms->type = "ceasefire";
                terms->details = "72-hour ceasefire and de-escalation";
            }
        } else if (strategy == Strategy::TransferWeapons) {
            *target_country_id = opportunistic_target_id == 0 ? primary_threat_id : opportunistic_target_id;
            terms->type = "weapons_transfer";
            terms->details = "Transfer reserve munitions to shape balance";
        } else if (strategy == Strategy::FocusEconomy) {
            terms->type = "economic_recovery";
            terms->details = "Shift budget to domestic resilience";
        } else if (strategy == Strategy::DevelopTechnology) {
            terms->type = "research_surge";
            terms->details = "Accelerate missile defense and cyber R&D";
        } else if (strategy == Strategy::FormAlliance) {
            *target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : opportunistic_target_id;
            terms->type = "alliance";
            terms->details = "Offer binding mutual security pact";
        } else if (strategy == Strategy::Betray) {
            *target_country_id = opportunistic_target_id == 0 ? primary_threat_id : opportunistic_target_id;
            terms->type = "betray";
            terms->details = "Exploit alliance weakness";
        } else if (strategy == Strategy::CyberOperation) {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : embargo_target_id;
            terms->type = "cyber_offense";
            terms->details = "Disrupt logistics and C2 networks";
        } else if (strategy == Strategy::SignTradeAgreement) {
            *target_country_id = best_trade_target_id;
            terms->type = "trade";
            terms->details = "Offer bilateral trade normalization";
        } else if (strategy == Strategy::CancelTradeAgreement) {
            *target_country_id = !self->trade_partner_ids.empty() ? self->trade_partner_ids.front() : best_trade_target_id;
            terms->type = "trade_cancel";
            terms->details = "Withdraw from unfavorable trade exposure";
        } else if (strategy == Strategy::ImposeEmbargo) {
            *target_country_id = embargo_target_id;
            terms->type = "embargo";
            terms->details = "Restrict critical imports and shipping access";
        } else if (strategy == Strategy::InvestInResourceExtraction) {
            terms->type = "resource_extraction";
            terms->details = "Redirect capital into domestic extraction";
        } else if (strategy == Strategy::ReduceMilitaryUpkeep) {
            terms->type = "demobilize";
            terms->details = "Reduce force posture to stabilize finances";
        } else if (strategy == Strategy::SuppressDissent) {
            terms->type = "internal_security";
            terms->details = "Contain unrest before it escalates";
        } else if (strategy == Strategy::HoldElections) {
            terms->type = "election";
            terms->details = "Seek renewed legitimacy before instability deepens";
        } else if (strategy == Strategy::CoupAttempt) {
            terms->type = "coup";
            terms->details = "Attempt emergency seizure of power";
        } else if (strategy == Strategy::ProposeDefensePact) {
            *target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : opportunistic_target_id;
            terms->type = "defense_pact";
            terms->details = "Offer mutual defense against external attack";
        } else if (strategy == Strategy::ProposeNonAggression) {
            *target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : primary_threat_id;
            terms->type = "non_aggression";
            terms->details = "Offer fixed-term non-aggression accord";
        } else if (strategy == Strategy::BreakTreaty) {
            *target_country_id = !self->non_aggression_pact_ids.empty() ? self->non_aggression_pact_ids.front() :
                (!self->defense_pact_ids.empty() ? self->defense_pact_ids.front() : best_treaty_target_id);
            terms->type = "break_treaty";
            terms->details = "Renounce binding commitments despite diplomatic cost";
        } else if (strategy == Strategy::RequestIntel) {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : embargo_target_id;
            terms->type = "intel_request";
            terms->details = "Expand reconnaissance and target libraries";
        } else if (strategy == Strategy::DeployUnits) {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : opportunistic_target_id;
            terms->type = "deploy_units";
            terms->details = "Shift ready formations and logistics toward the frontier";
        } else if (strategy == Strategy::TacticalNuke) {
            *target_country_id = primary_threat_id;
            terms->type = "tactical_nuclear_option";
            terms->details = "Limited strike against front-line formations";
        } else if (strategy == Strategy::StrategicNuke) {
            *target_country_id = primary_threat_id;
            terms->type = "strategic_nuclear_option";
            terms->details = "Countervalue strike with regime-level consequences";
        } else if (strategy == Strategy::CyberAttack) {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : embargo_target_id;
            terms->type = "cyber_attack";
            terms->details = "Attack supply networks and command systems";
        }
    };

    auto requires_target = [](Strategy strategy) {
        return strategy == Strategy::Attack || strategy == Strategy::Negotiate || strategy == Strategy::TransferWeapons ||
               strategy == Strategy::FormAlliance || strategy == Strategy::Betray || strategy == Strategy::CyberOperation ||
               strategy == Strategy::SignTradeAgreement || strategy == Strategy::CancelTradeAgreement || strategy == Strategy::ImposeEmbargo ||
               strategy == Strategy::ProposeDefensePact || strategy == Strategy::ProposeNonAggression || strategy == Strategy::BreakTreaty ||
               strategy == Strategy::RequestIntel || strategy == Strategy::DeployUnits || strategy == Strategy::TacticalNuke ||
               strategy == Strategy::StrategicNuke || strategy == Strategy::CyberAttack;
    };

    auto ensure_target = [&](Strategy strategy, uint16_t* target_country_id) {
        if (target_country_id == nullptr || !requires_target(strategy) || *target_country_id != 0) {
            return;
        }
        if (strategy == Strategy::SignTradeAgreement || strategy == Strategy::CancelTradeAgreement) {
            *target_country_id = best_trade_target_id != 0 ? best_trade_target_id : best_treaty_target_id;
        } else if (strategy == Strategy::ProposeDefensePact || strategy == Strategy::ProposeNonAggression || strategy == Strategy::Negotiate || strategy == Strategy::FormAlliance) {
            *target_country_id = best_treaty_target_id != 0 ? best_treaty_target_id : primary_threat_id;
        } else {
            *target_country_id = primary_threat_id != 0 ? primary_threat_id : (best_trade_target_id != 0 ? best_trade_target_id : embargo_target_id);
        }
    };

    populate_action(decision.strategy, &decision.target_country_id, &decision.terms);
    ensure_target(decision.strategy, &decision.target_country_id);

    if (coup_risk_norm > 0.85f && decision.strategy == Strategy::Attack) {
        decision.strategy = Strategy::SuppressDissent;
        decision.target_country_id = 0;
        decision.terms = {};
        populate_action(decision.strategy, &decision.target_country_id, &decision.terms);
        decision.terms.details = "Stabilize regime before escalating abroad";
    } else if (election_urgency_norm > 0.8f && war_weariness_norm > 0.55f && decision.strategy == Strategy::Attack) {
        decision.strategy = Strategy::HoldElections;
        decision.target_country_id = 0;
        decision.terms = {};
        populate_action(decision.strategy, &decision.target_country_id, &decision.terms);
        decision.terms.details = "Reset legitimacy before continuing conflict";
    } else if ((decision.strategy == Strategy::StrategicNuke || decision.strategy == Strategy::TacticalNuke) &&
               (nuclear_norm < 0.45f || (strongest_target_second_strike && self->second_strike_capable == false))) {
        decision.strategy = Strategy::RequestIntel;
        decision.target_country_id = 0;
        decision.terms = {};
        populate_action(decision.strategy, &decision.target_country_id, &decision.terms);
        decision.terms.details = "Refuse nuclear use without credible targeting and survivability";
    }

    auto score_of = [&](Strategy strategy) -> float {
        const size_t idx = static_cast<size_t>(strategy);
        return idx < action_count ? logits.data[idx] : -1e9f;
    };
    auto best_of = [&](std::initializer_list<Strategy> strategies) -> Strategy {
        Strategy best_strategy = Strategy::Defend;
        float best_strategy_score = -1e9f;
        for (Strategy strategy : strategies) {
            if (strategy == decision.strategy) {
                continue;
            }
            const float score = score_of(strategy);
            if (score > best_strategy_score) {
                best_strategy_score = score;
                best_strategy = strategy;
            }
        }
        return best_strategy;
    };

    if (strategic_depth_norm > 0.28f) {
        Strategy secondary_strategy = Strategy::Defend;
        if (decision.strategy == Strategy::Attack || decision.strategy == Strategy::DeployUnits ||
            decision.strategy == Strategy::CyberAttack || decision.strategy == Strategy::CyberOperation ||
            decision.strategy == Strategy::TacticalNuke || decision.strategy == Strategy::StrategicNuke) {
            if (intel_opportunity > model_tuning::kSecondaryIntelOpportunityThresholdAggressive && primary_threat_id != 0) {
                secondary_strategy = best_of({Strategy::RequestIntel, Strategy::CyberOperation});
            } else if (neighbor_trust_high > 0.54f && best_treaty_target_id != 0) {
                secondary_strategy = best_of({Strategy::ProposeDefensePact, Strategy::ProposeNonAggression, Strategy::Negotiate});
            }
        } else if (decision.strategy == Strategy::Negotiate || decision.strategy == Strategy::SignTradeAgreement ||
                   decision.strategy == Strategy::ProposeDefensePact || decision.strategy == Strategy::ProposeNonAggression ||
                   decision.strategy == Strategy::FormAlliance || decision.strategy == Strategy::FocusEconomy ||
                   decision.strategy == Strategy::DevelopTechnology) {
            if (threat_ratio > 0.92f && primary_threat_id != 0) {
                secondary_strategy = best_of({Strategy::DeployUnits, Strategy::RequestIntel});
            } else if (crisis_pressure > 0.58f) {
                secondary_strategy = best_of({Strategy::FocusEconomy, Strategy::SuppressDissent});
            }
        } else if (decision.strategy == Strategy::Defend) {
            if (intel_opportunity > 0.30f && primary_threat_id != 0) {
                secondary_strategy = best_of({Strategy::RequestIntel, Strategy::DeployUnits});
            } else if (best_treaty_target_id != 0 && neighbor_trust_high > 0.50f) {
                secondary_strategy = best_of({Strategy::ProposeDefensePact, Strategy::ProposeNonAggression});
            }
        }

        if (secondary_strategy != Strategy::Defend) {
            decision.has_secondary_action = true;
            decision.secondary_action.strategy = secondary_strategy;
            decision.secondary_action.commitment = std::clamp(0.18f + strategic_depth_norm * 0.30f + opponent_confidence_avg * 0.16f - crisis_pressure * 0.12f,
                                                             0.15f,
                                                             0.70f);
            populate_action(decision.secondary_action.strategy,
                            &decision.secondary_action.target_country_id,
                            &decision.secondary_action.terms);
            ensure_target(decision.secondary_action.strategy, &decision.secondary_action.target_country_id);
            if ((decision.secondary_action.strategy == Strategy::RequestIntel || decision.secondary_action.strategy == Strategy::CyberOperation) &&
                decision.secondary_action.target_country_id == decision.target_country_id && decision.target_country_id == 0) {
                decision.has_secondary_action = false;
                decision.secondary_action = {};
            }
        }
    }
    return decision;
}

size_t Model::input_dim() const {
    if (layers_.empty()) {
        return 0;
    }
    if (layers_.front().weight.shape.size() != 2) {
        return 0;
    }
    return layers_.front().weight.shape[1];
}

size_t Model::output_dim() const {
    if (layers_.empty()) {
        return 0;
    }
    if (layers_.back().weight.shape.size() != 2) {
        return 0;
    }
    return layers_.back().weight.shape[0];
}