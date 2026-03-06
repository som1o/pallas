#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Tensor weight;
    Tensor bias;
    size_t in_dim, out_dim;

    Linear(size_t in, size_t out);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output, const Tensor& input);
};

#endif