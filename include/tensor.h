#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <vector>
#include <cstddef>

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;

    Tensor(std::vector<size_t> s, float init = 0.0f);
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;
    void zero_grad();
    void disable_grad_storage();
    void enable_grad_storage();
    [[nodiscard]] bool grad_enabled() const;
    [[nodiscard]] static Tensor matmul(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor add(const Tensor& a, const Tensor& b);
    [[nodiscard]] Tensor transpose() const;
    void softmax();
};

[[nodiscard]] float cross_entropy(const Tensor& logits, uint32_t target);
[[nodiscard]] Tensor grad_cross_entropy(const Tensor& logits, uint32_t target);
[[nodiscard]] float cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight);
[[nodiscard]] Tensor grad_cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight);
[[nodiscard]] bool top_k_hit(const Tensor& logits, uint32_t target, size_t k);

#endif