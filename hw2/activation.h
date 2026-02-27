#pragma once
#include "tensor.h"

class ReLU {
public:
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

private:
    Tensor input_;  // 保存前向传播的输入，反向传播时需要
};

class Sigmoid {
public:
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

private:
    Tensor output_;  // 保存前向传播的输出，反向传播时需要
};
