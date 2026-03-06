#include "tensor.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Tensor::Tensor(std::vector<size_t> s, float init) : shape(s) {
    size_t size = 1;
    for (auto d : s) size *= d;
    data.assign(size, init);
    grad.assign(size, 0.0f);
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::disable_grad_storage() {
    grad.clear();
    grad.shrink_to_fit();
}

void Tensor::enable_grad_storage() {
    grad.assign(data.size(), 0.0f);
}

bool Tensor::grad_enabled() const {
    return grad.size() == data.size();
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("matmul expects 2D tensors");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("matmul shape mismatch");
    }

    size_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        float* c_row = &c.data[i * n];
        for (size_t p = 0; p < k; p++) {
            const float a_ip = a.data[i * k + p];
            const float* b_row = &b.data[p * n];
            for (size_t j = 0; j < n; j++) {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
    return c;
}

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("add shape mismatch");
    }

    Tensor c(a.shape, 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < a.data.size(); i++) {
        c.data[i] = a.data[i] + b.data[i];
    }
    return c;
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::runtime_error("transpose expects 2D tensor");
    }

    size_t rows = shape[0], cols = shape[1];
    Tensor t({cols, rows}, 0.0f);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            t.data[j * rows + i] = data[i * cols + j];
        }
    }
    return t;
}

void Tensor::softmax() {
    if (shape.empty()) {
        throw std::runtime_error("softmax expects non-empty shape");
    }

    size_t batch = 1;
    size_t vocab = 0;
    if (shape.size() == 1) {
        vocab = shape[0];
    } else if (shape.size() == 2) {
        batch = shape[0];
        vocab = shape[1];
    } else {
        throw std::runtime_error("softmax expects 1D or 2D tensor");
    }

    if (vocab == 0 || data.size() != batch * vocab) {
        throw std::runtime_error("softmax shape/data mismatch");
    }

    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        float max_val = *std::max_element(data.begin() + b * vocab, data.begin() + (b + 1) * vocab);
        float sum = 0.0f;
        for (size_t v = 0; v < vocab; v++) {
            data[b * vocab + v] = exp(data[b * vocab + v] - max_val);
            sum += data[b * vocab + v];
        }
        for (size_t v = 0; v < vocab; v++) {
            data[b * vocab + v] /= sum;
        }
    }
}

float cross_entropy(const Tensor& logits, uint32_t target) {
    if (logits.data.empty()) {
        return 0.0f;
    }
    const size_t classes = logits.shape.size() == 1 ? logits.shape[0] : (logits.shape.size() == 2 ? logits.shape[1] : 0);
    if (classes == 0 || target >= classes || target >= logits.data.size()) {
        return 0.0f;
    }
    Tensor probs = logits;
    probs.softmax();
    const float p = std::max(probs.data[target], std::numeric_limits<float>::min());
    return -std::log(p);
}

Tensor grad_cross_entropy(const Tensor& logits, uint32_t target) {
    Tensor grad = logits;
    grad.softmax();
    if (target < grad.data.size()) {
        grad.data[target] -= 1.0f;
    } else {
        std::fill(grad.data.begin(), grad.data.end(), 0.0f);
    }
    return grad;
}

float cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight) {
    Tensor probs = logits;
    probs.softmax();

    const size_t classes = logits.shape.size() == 1 ? logits.shape[0] : (logits.shape.size() == 2 ? logits.shape[1] : 0);
    if (target >= classes) {
        return 0.0f;
    }

    label_smoothing = std::min(0.999f, std::max(0.0f, label_smoothing));
    const float smooth_other = classes > 1 ? label_smoothing / static_cast<float>(classes - 1) : 0.0f;
    const float smooth_target = 1.0f - label_smoothing;

    float loss = 0.0f;
    for (size_t c = 0; c < classes; ++c) {
        const float p = std::max(probs.data[c], std::numeric_limits<float>::min());
        const float y = (c == target) ? smooth_target : smooth_other;
        loss -= y * std::log(p);
    }
    return target_weight * loss;
}

Tensor grad_cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight) {
    Tensor grad = logits;
    grad.softmax();

    const size_t classes = logits.shape.size() == 1 ? logits.shape[0] : (logits.shape.size() == 2 ? logits.shape[1] : 0);
    if (target >= classes) {
        std::fill(grad.data.begin(), grad.data.end(), 0.0f);
        return grad;
    }

    label_smoothing = std::min(0.999f, std::max(0.0f, label_smoothing));
    const float smooth_other = classes > 1 ? label_smoothing / static_cast<float>(classes - 1) : 0.0f;
    const float smooth_target = 1.0f - label_smoothing;

    for (size_t c = 0; c < classes; ++c) {
        const float y = (c == target) ? smooth_target : smooth_other;
        grad.data[c] = (grad.data[c] - y) * target_weight;
    }
    return grad;
}

bool top_k_hit(const Tensor& logits, uint32_t target, size_t k) {
    const size_t classes = logits.shape.size() == 1 ? logits.shape[0] : (logits.shape.size() == 2 ? logits.shape[1] : 0);
    if (target >= classes) {
        return false;
    }
    if (k >= classes) {
        return true;
    }

    std::vector<size_t> idx(classes);
    for (size_t i = 0; i < classes; ++i) {
        idx[i] = i;
    }

    std::partial_sort(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [&](size_t a, size_t b) { return logits.data[a] > logits.data[b]; });

    for (size_t i = 0; i < k; ++i) {
        if (idx[i] == target) {
            return true;
        }
    }
    return false;
}