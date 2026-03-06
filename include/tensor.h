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
    void zero_grad();
    void disable_grad_storage();
    void enable_grad_storage();
    bool grad_enabled() const;
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor add(const Tensor& a, const Tensor& b);
    Tensor transpose() const;
    void softmax();
};

float cross_entropy(const Tensor& logits, uint32_t target);
Tensor grad_cross_entropy(const Tensor& logits, uint32_t target);
float cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight);
Tensor grad_cross_entropy_advanced(const Tensor& logits, uint32_t target, float label_smoothing, float target_weight);
bool top_k_hit(const Tensor& logits, uint32_t target, size_t k);

#endif