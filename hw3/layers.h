#pragma once
#include <cstddef>

// ========== 全连接层 ==========
// input:  [N, C_in]   weight: [C_out, C_in]   bias: [C_out]   output: [N, C_out]
void fc_forward(const float *input, const float *weight, const float *bias, float *output, size_t N,
    size_t C_in, size_t C_out);
void fc_backward_input(const float *grad_output, const float *weight, float *grad_input, size_t N,
    size_t C_in, size_t C_out);
void fc_backward_weight(const float *grad_output, const float *input, float *grad_weight, size_t N,
    size_t C_in, size_t C_out);
void fc_backward_bias(const float *grad_output, float *grad_bias, size_t N, size_t C_out);

// ========== 池化层（2x2 Max Pooling, stride=2）==========
// input:  [N, C, H, W]   output: [N, C, H/2, W/2]
// mask:   [N, C, H/2, W/2]，存储最大值在 input 中的 flat index，重复值取左上角
void pool_forward(const float *input, float *output, int *mask, size_t N, size_t C, size_t H,
    size_t W);
void pool_backward(const float *grad_output, const int *mask, float *grad_input, size_t N,
    size_t C, size_t H, size_t W);

// ========== SoftMax ==========
// input/output: [N, C]
void softmax_forward(const float *input, float *output, size_t N, size_t C);

// ========== Cross Entropy Loss ==========
// probs: [N, C]   labels: [N]（类别索引，存为 float）
float cross_entropy_forward(const float *probs, const float *labels, size_t N, size_t C);
void cross_entropy_backward(
    const float *probs, const float *labels, float *grad, size_t N, size_t C);

// ========== 卷积层（3x3, padding=1, stride=1）==========
// input:  [N, C_in, H, W]   weight: [C_out, C_in, 3, 3]   output: [N, C_out, H, W]
void conv_forward(const float *input, const float *weight, const float *bias, float *output,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W);
void conv_backward_input(const float *grad_output, const float *weight, float *grad_input, size_t N,
    size_t C_in, size_t C_out, size_t H, size_t W);
void conv_backward_weight(const float *grad_output, const float *input, float *grad_weight,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W);
