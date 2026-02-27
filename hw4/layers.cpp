#include "layers.h"
#include "gemm.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ========== 全连接层 ==========

void fc_forward(const float *input, const float *weight, const float *bias, float *output, size_t N,
    size_t C_in, size_t C_out) {
    // output = input * weight^T + bias
    // input(N×C_in) * weight^T(C_in×C_out) = output(N×C_out)
    gemm_gpu(false, true, N, C_out, C_in, 1.0f, input, weight, 0.0f, output);

    // ones[N,1] * bias[1,C_out] 广播加 bias
    std::vector<float> ones(N, 1.0f);
    gemm_gpu(false, false, N, C_out, 1, 1.0f, ones.data(), bias, 1.0f, output);
}

void fc_backward_input(const float *grad_output, const float *weight, float *grad_input, size_t N,
    size_t C_in, size_t C_out) {
    // dX = dY * W, [N,C_out] * [C_out,C_in] = [N,C_in]
    gemm_gpu(false, false, N, C_in, C_out, 1.0f, grad_output, weight, 0.0f, grad_input);
}

void fc_backward_weight(const float *grad_output, const float *input, float *grad_weight, size_t N,
    size_t C_in, size_t C_out) {
    // dW = dY^T * X, [C_out,N] * [N,C_in] = [C_out,C_in]
    gemm_gpu(true, false, C_out, C_in, N, 1.0f, grad_output, input, 0.0f, grad_weight);
}

void fc_backward_bias(const float *grad_output, float *grad_bias, size_t N, size_t C_out) {
    // db = sum(dY, axis=0), 对 N 维求和
    for (size_t j = 0; j < C_out; ++j) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; ++n)
            sum += grad_output[n * C_out + j];
        grad_bias[j] = sum;
    }
}

// ========== 池化层 ==========

void pool_forward(const float *input, float *output, int *mask, size_t N, size_t C, size_t H,
    size_t W) {
    size_t H_out = H / 2, W_out = W / 2;
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t i = 0; i < H_out; ++i)
                for (size_t j = 0; j < W_out; ++j) {
                    float max_val = -1e38f;
                    int max_idx = -1;
                    for (size_t di = 0; di < 2; ++di)
                        for (size_t dj = 0; dj < 2; ++dj) {
                            int idx = ((n * C + c) * H + i * 2 + di) * W + j * 2 + dj;
                            if (input[idx] > max_val) { max_val = input[idx]; max_idx = idx; }
                        }
                    size_t out_idx = ((n * C + c) * H_out + i) * W_out + j;
                    output[out_idx] = max_val;
                    mask[out_idx] = max_idx;
                }
}

void pool_backward(const float *grad_output, const int *mask, float *grad_input, size_t N,
    size_t C, size_t H, size_t W) {
    size_t H_out = H / 2, W_out = W / 2;
    std::fill(grad_input, grad_input + N * C * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t i = 0; i < H_out; ++i)
                for (size_t j = 0; j < W_out; ++j) {
                    size_t out_idx = ((n * C + c) * H_out + i) * W_out + j;
                    grad_input[mask[out_idx]] += grad_output[out_idx];
                }
}

// ========== SoftMax ==========

void softmax_forward(const float *input, float *output, size_t N, size_t C) {
    for (size_t n = 0; n < N; ++n) {
        const float *in = input + n * C;
        float *out = output + n * C;
        // 数值稳定：减去最大值
        float max_val = *std::max_element(in, in + C);
        float sum = 0.0f;
        for (size_t c = 0; c < C; ++c) { out[c] = std::exp(in[c] - max_val); sum += out[c]; }
        for (size_t c = 0; c < C; ++c) out[c] /= sum;
    }
}

// ========== Cross Entropy Loss ==========

float cross_entropy_forward(const float *probs, const float *labels, size_t N, size_t C) {
    float loss = 0.0f;
    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels[n]);
        loss -= std::log(probs[n * C + label] + 1e-12f);
    }
    return loss / N;
}

void cross_entropy_backward(
    const float *probs, const float *labels, float *grad, size_t N, size_t C) {
    // softmax + cross entropy 合并梯度：dL/dz_i = (p_i - 1_{i==y}) / N
    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels[n]);
        for (size_t c = 0; c < C; ++c)
            grad[n * C + c] = probs[n * C + c] / N;
        grad[n * C + label] -= 1.0f / N;
    }
}

// ========== 卷积层 ==========

void conv_forward(const float *input, const float *weight, const float *bias, float *output,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    std::vector<float> col(C_in * 9 * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        // im2col
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            float val = (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W)
                                ? input[((n * C_in + c) * H + h_in) * W + w_in] : 0.0f;
                            size_t k = c * 9 + kh * 3 + kw;
                            col[k * H * W + i * W + j] = val;
                        }
        float *out_n = output + n * C_out * H * W;
        gemm_gpu(false, false, C_out, H * W, C_in * 9, 1.0f, weight, col.data(), 0.0f, out_n);
        for (size_t c = 0; c < C_out; ++c)
            for (size_t hw = 0; hw < H * W; ++hw)
                out_n[c * H * W + hw] += bias[c];
    }
}

void conv_backward_input(const float *grad_output, const float *weight, float *grad_input,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    std::vector<float> col(C_in * 9 * H * W, 0.0f);
    std::fill(grad_input, grad_input + N * C_in * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        const float *dY_n = grad_output + n * C_out * H * W;
        // dcol = weight^T * dY_n: [C_in*9, H*W]
        gemm_gpu(true, false, C_in * 9, H * W, C_out, 1.0f, weight, dY_n, 0.0f, col.data());
        // col2im
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            if (h_in < 0 || h_in >= (int)H || w_in < 0 || w_in >= (int)W) continue;
                            size_t k = c * 9 + kh * 3 + kw;
                            grad_input[((n * C_in + c) * H + h_in) * W + w_in] +=
                                col[k * H * W + i * W + j];
                        }
    }
}

void conv_backward_weight(const float *grad_output, const float *input, float *grad_weight,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    std::vector<float> col(C_in * 9 * H * W, 0.0f);
    std::fill(grad_weight, grad_weight + C_out * C_in * 9, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        // im2col
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            float val = (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W)
                                ? input[((n * C_in + c) * H + h_in) * W + w_in] : 0.0f;
                            size_t k = c * 9 + kh * 3 + kw;
                            col[k * H * W + i * W + j] = val;
                        }
        const float *dY_n = grad_output + n * C_out * H * W;
        float beta = (n == 0) ? 0.0f : 1.0f;
        gemm_gpu(false, true, C_out, C_in * 9, H * W, 1.0f, dY_n, col.data(), beta, grad_weight);
    }
}
