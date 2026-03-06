#include "linear.h"

#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <stdexcept>

namespace {

std::mutex g_linear_rng_mu;

std::mt19937& linear_rng() {
    static std::mt19937 rng([]() {
        const char* env = std::getenv("PALLAS_LINEAR_SEED");
        if (env != nullptr) {
            try {
                return static_cast<uint32_t>(std::stoul(env));
            } catch (...) {
            }
        }
        return 1337U;
    }());
    return rng;
}

}  // namespace

Linear::Linear(size_t in, size_t out) : in_dim(in), out_dim(out), weight({out, in}, 0.0f), bias({out}, 0.0f) {
    std::lock_guard<std::mutex> lock(g_linear_rng_mu);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& w : weight.data) {
        w = dist(linear_rng());
    }
    for (auto& b : bias.data) {
        b = dist(linear_rng());
    }
}

void Linear::set_global_seed(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_linear_rng_mu);
    linear_rng().seed(seed);
}

Tensor Linear::forward(const Tensor& input) {
    if (input.shape.size() != 2 || input.shape[1] != in_dim) {
        throw std::runtime_error("Linear::forward input shape mismatch");
    }
    const size_t batch = input.shape[0];
    Tensor out({batch, out_dim}, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        const float* in_row = &input.data[b * in_dim];
        float* out_row = &out.data[b * out_dim];
        for (size_t o = 0; o < out_dim; ++o) {
            const float* w_row = &weight.data[o * in_dim];
            float sum = bias.data[o];
            for (size_t i = 0; i < in_dim; ++i) {
                sum += in_row[i] * w_row[i];
            }
            out_row[o] = sum;
        }
    }
    return out;
}

Tensor Linear::backward(const Tensor& grad_output, const Tensor& input) {
    Tensor gt = grad_output.transpose();
    Tensor grad_w = Tensor::matmul(gt, input);
    Tensor grad_b({out_dim}, 0.0f);
    for (size_t b = 0; b < grad_output.shape[0]; b++) {
        for (size_t o = 0; o < out_dim; o++) {
            grad_b.data[o] += grad_output.data[b * out_dim + o];
        }
    }
    for (size_t i = 0; i < weight.grad.size(); i++) {
        weight.grad[i] += grad_w.data[i];
    }
    for (size_t i = 0; i < bias.grad.size(); i++) {
        bias.grad[i] += grad_b.data[i];
    }
    Tensor grad_in = Tensor::matmul(grad_output, weight);
    return grad_in;
}