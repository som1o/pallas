#include "linear.h"
#include <random>

Linear::Linear(size_t in, size_t out) : in_dim(in), out_dim(out), weight({out, in}, 0.0f), bias({out}, 0.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& w : weight.data) w = dist(gen);
    for (auto& b : bias.data) b = dist(gen);
}

Tensor Linear::forward(const Tensor& input) {
    Tensor wt = weight.transpose();
    Tensor out = Tensor::matmul(input, wt);
    for (size_t b = 0; b < input.shape[0]; b++) {
        for (size_t o = 0; o < out_dim; o++) {
            out.data[b * out_dim + o] += bias.data[o];
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