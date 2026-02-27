#include "layers.h"
#include "gemm.h"
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>

// ===== FC kernels =====

__global__ void bias_add_kernel(float *output, const float *bias, int N, int C_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out) return;
    output[idx] += bias[idx % C_out];
}

__global__ void fill_ones_kernel(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 1.0f;
}

// ===== Pool kernels =====

__global__ void pool_fwd_kernel(const float *input, float *output, int *mask,
    int N, int C, int H, int W, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;
    int j = idx % W_out;
    int i = (idx / W_out) % H_out;
    int c = (idx / W_out / H_out) % C;
    int n = idx / W_out / H_out / C;
    float max_val = -1e38f;
    int max_idx = -1;
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj) {
            int in_idx = ((n * C + c) * H + i * 2 + di) * W + j * 2 + dj;
            if (input[in_idx] > max_val) { max_val = input[in_idx]; max_idx = in_idx; }
        }
    output[idx] = max_val;
    mask[idx] = max_idx;
}

__global__ void pool_bwd_kernel(const float *grad_output, const int *mask,
    float *grad_input, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    atomicAdd(&grad_input[mask[idx]], grad_output[idx]);
}

// ===== Softmax kernel =====
// 一个 block 处理一行，shared memory 做并行 reduction
// launch: softmax_kernel<<<N, 256, 256*sizeof(float)>>>
__global__ void softmax_kernel(const float *input, float *output, int N, int C) {
    int n = blockIdx.x;
    if (n >= N) return;
    extern __shared__ float smem[];
    const float *in = input + n * C;
    float *out = output + n * C;

    // Phase 1: 并行找最大值
    float max_val = -1e38f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        max_val = fmaxf(max_val, in[c]);
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Phase 2: 并行计算 exp 并求和
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        out[c] = expf(in[c] - max_val);
        sum += out[c];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Phase 3: 归一化
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        out[c] /= sum;
}

// ===== Cross entropy kernels =====

__global__ void ce_fwd_kernel(const float *probs, const float *labels,
    float *losses, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    losses[n] = -logf(probs[n * C + (int)labels[n]] + 1e-12f);
}

__global__ void ce_bwd_kernel(const float *probs, const float *labels,
    float *grad, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    int n = idx / C, c = idx % C;
    grad[idx] = probs[idx] / N;
    if (c == (int)labels[n]) grad[idx] -= 1.0f / N;
}

// ===== im2col / col2im kernels (3x3, padding=1, stride=1) =====
// col layout: [N, C_in*9, H*W]

__global__ void im2col_kernel(const float *input, float *col,
    int N, int C_in, int H, int W) {
    int total = N * C_in * 9 * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w_out = idx % W;
    int h_out = (idx / W) % H;
    int k     = (idx / W / H) % (C_in * 9);
    int n     = idx / W / H / (C_in * 9);
    int c = k / 9, kh = (k % 9) / 3, kw = k % 3;
    int h_in = h_out + kh - 1, w_in = w_out + kw - 1;
    float val = (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        ? input[((n * C_in + c) * H + h_in) * W + w_in] : 0.0f;
    col[(n * C_in * 9 + k) * H * W + h_out * W + w_out] = val;
}

__global__ void col2im_kernel(const float *col, float *input,
    int N, int C_in, int H, int W) {
    int total = N * C_in * 9 * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w_out = idx % W;
    int h_out = (idx / W) % H;
    int k     = (idx / W / H) % (C_in * 9);
    int n     = idx / W / H / (C_in * 9);
    int c = k / 9, kh = (k % 9) / 3, kw = k % 3;
    int h_in = h_out + kh - 1, w_in = w_out + kw - 1;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        atomicAdd(&input[((n * C_in + c) * H + h_in) * W + w_in],
            col[(n * C_in * 9 + k) * H * W + h_out * W + w_out]);
}

__global__ void conv_bias_add_kernel(float *output, const float *bias,
    int N, int C_out, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * HW) return;
    output[idx] += bias[(idx / HW) % C_out];
}

#endif // __CUDACC__

// ========== 全连接层 ==========

void fc_forward(const float *input, const float *weight, const float *bias, float *output,
    size_t N, size_t C_in, size_t C_out) {
    gemm_gpu(false, true, N, C_out, C_in, 1.0f, input, weight, 0.0f, output);
#ifdef __CUDACC__
    int total = N * C_out;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(output, bias, N, C_out);
#else
    std::vector<float> ones(N, 1.0f);
    gemm_gpu(false, false, N, C_out, 1, 1.0f, ones.data(), bias, 1.0f, output);
#endif
}

void fc_backward_input(const float *grad_output, const float *weight, float *grad_input,
    size_t N, size_t C_in, size_t C_out) {
    // dX = dY * W, [N,C_out] * [C_out,C_in] = [N,C_in]
    gemm_gpu(false, false, N, C_in, C_out, 1.0f, grad_output, weight, 0.0f, grad_input);
}

void fc_backward_weight(const float *grad_output, const float *input, float *grad_weight,
    size_t N, size_t C_in, size_t C_out) {
    // dW = dY^T * X, [C_out,N] * [N,C_in] = [C_out,C_in]
    gemm_gpu(true, false, C_out, C_in, N, 1.0f, grad_output, input, 0.0f, grad_weight);
}

void fc_backward_bias(const float *grad_output, float *grad_bias, size_t N, size_t C_out) {
#ifdef __CUDACC__
    // db = grad_output^T * ones[N,1]，用 gemm 代替自定义 reduction kernel
    float *ones_d;
    cudaMalloc(&ones_d, N * sizeof(float));
    fill_ones_kernel<<<(N + 255) / 256, 256>>>(ones_d, N);
    gemm_gpu(true, false, C_out, 1, N, 1.0f, grad_output, ones_d, 0.0f, grad_bias);
    cudaFree(ones_d);
#else
    for (size_t j = 0; j < C_out; ++j) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; ++n) sum += grad_output[n * C_out + j];
        grad_bias[j] = sum;
    }
#endif
}

// ========== 池化层 ==========

void pool_forward(const float *input, float *output, int *mask, size_t N, size_t C,
    size_t H, size_t W) {
    size_t H_out = H / 2, W_out = W / 2;
#ifdef __CUDACC__
    int total = N * C * H_out * W_out;
    pool_fwd_kernel<<<(total + 255) / 256, 256>>>(
        input, output, mask, N, C, H, W, H_out, W_out);
#else
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
#endif
}

void pool_backward(const float *grad_output, const int *mask, float *grad_input,
    size_t N, size_t C, size_t H, size_t W) {
    size_t H_out = H / 2, W_out = W / 2;
    int total_out = N * C * H_out * W_out;
#ifdef __CUDACC__
    cudaMemset(grad_input, 0, N * C * H * W * sizeof(float));
    pool_bwd_kernel<<<(total_out + 255) / 256, 256>>>(grad_output, mask, grad_input, total_out);
#else
    std::fill(grad_input, grad_input + N * C * H * W, 0.0f);
    for (int i = 0; i < total_out; ++i) grad_input[mask[i]] += grad_output[i];
#endif
}

// ========== SoftMax ==========

void softmax_forward(const float *input, float *output, size_t N, size_t C) {
#ifdef __CUDACC__
    // 一个 block 处理一行，shared memory 大小 = blockDim.x * sizeof(float)
    softmax_kernel<<<N, 256, 256 * sizeof(float)>>>(input, output, N, C);
#else
    for (size_t n = 0; n < N; ++n) {
        const float *in = input + n * C;
        float *out = output + n * C;
        float max_val = *std::max_element(in, in + C);
        float sum = 0.0f;
        for (size_t c = 0; c < C; ++c) { out[c] = std::exp(in[c] - max_val); sum += out[c]; }
        for (size_t c = 0; c < C; ++c) out[c] /= sum;
    }
#endif
}

// ========== Cross Entropy Loss ==========

float cross_entropy_forward(const float *probs, const float *labels, size_t N, size_t C) {
#ifdef __CUDACC__
    float *losses_d;
    cudaMalloc(&losses_d, N * sizeof(float));
    ce_fwd_kernel<<<(N + 255) / 256, 256>>>(probs, labels, losses_d, N, C);
    std::vector<float> losses_h(N);
    cudaMemcpy(losses_h.data(), losses_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(losses_d);
    float total = 0.0f;
    for (float v : losses_h) total += v;
    return total / N;
#else
    float loss = 0.0f;
    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels[n]);
        loss -= std::log(probs[n * C + label] + 1e-12f);
    }
    return loss / N;
#endif
}

void cross_entropy_backward(const float *probs, const float *labels, float *grad,
    size_t N, size_t C) {
#ifdef __CUDACC__
    int total = N * C;
    ce_bwd_kernel<<<(total + 255) / 256, 256>>>(probs, labels, grad, N, C);
#else
    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels[n]);
        for (size_t c = 0; c < C; ++c) grad[n * C + c] = probs[n * C + c] / N;
        grad[n * C + label] -= 1.0f / N;
    }
#endif
}

// ========== 卷积层 ==========

void conv_forward(const float *input, const float *weight, const float *bias, float *output,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    size_t col_size = N * C_in * 9 * H * W;
#ifdef __CUDACC__
    float *col;
    cudaMalloc(&col, col_size * sizeof(float));
    im2col_kernel<<<(col_size + 255) / 256, 256>>>(input, col, N, C_in, H, W);
    for (size_t n = 0; n < N; ++n) {
        const float *col_n = col + n * C_in * 9 * H * W;
        float *out_n = output + n * C_out * H * W;
        gemm_gpu(false, false, C_out, H * W, C_in * 9, 1.0f, weight, col_n, 0.0f, out_n);
    }
    int total_out = N * C_out * H * W;
    conv_bias_add_kernel<<<(total_out + 255) / 256, 256>>>(output, bias, N, C_out, H * W);
    cudaFree(col);
#else
    std::vector<float> col(col_size, 0.0f);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            float val = (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W)
                                ? input[((n * C_in + c) * H + h_in) * W + w_in] : 0.0f;
                            size_t k = c * 9 + kh * 3 + kw;
                            col[(n * C_in * 9 + k) * H * W + i * W + j] = val;
                        }
    for (size_t n = 0; n < N; ++n) {
        const float *col_n = col.data() + n * C_in * 9 * H * W;
        float *out_n = output + n * C_out * H * W;
        gemm_gpu(false, false, C_out, H * W, C_in * 9, 1.0f, weight, col_n, 0.0f, out_n);
        for (size_t c = 0; c < C_out; ++c)
            for (size_t hw = 0; hw < H * W; ++hw)
                out_n[c * H * W + hw] += bias[c];
    }
#endif
}

void conv_backward_input(const float *grad_output, const float *weight, float *grad_input,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    size_t col_size = N * C_in * 9 * H * W;
#ifdef __CUDACC__
    float *col;
    cudaMalloc(&col, col_size * sizeof(float));
    cudaMemset(grad_input, 0, N * C_in * H * W * sizeof(float));
    // dcol[n] = weight^T * dY[n]: [C_in*9, H*W]
    for (size_t n = 0; n < N; ++n) {
        float *col_n = col + n * C_in * 9 * H * W;
        const float *dY_n = grad_output + n * C_out * H * W;
        gemm_gpu(true, false, C_in * 9, H * W, C_out, 1.0f, weight, dY_n, 0.0f, col_n);
    }
    col2im_kernel<<<(col_size + 255) / 256, 256>>>(col, grad_input, N, C_in, H, W);
    cudaFree(col);
#else
    std::vector<float> col(col_size, 0.0f);
    std::fill(grad_input, grad_input + N * C_in * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        float *col_n = col.data() + n * C_in * 9 * H * W;
        const float *dY_n = grad_output + n * C_out * H * W;
        gemm_gpu(true, false, C_in * 9, H * W, C_out, 1.0f, weight, dY_n, 0.0f, col_n);
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            if (h_in < 0 || h_in >= (int)H || w_in < 0 || w_in >= (int)W) continue;
                            size_t k = c * 9 + kh * 3 + kw;
                            grad_input[((n * C_in + c) * H + h_in) * W + w_in] +=
                                col_n[k * H * W + i * W + j];
                        }
    }
#endif
}

void conv_backward_weight(const float *grad_output, const float *input, float *grad_weight,
    size_t N, size_t C_in, size_t C_out, size_t H, size_t W) {
    size_t col_size = N * C_in * 9 * H * W;
#ifdef __CUDACC__
    float *col;
    cudaMalloc(&col, col_size * sizeof(float));
    im2col_kernel<<<(col_size + 255) / 256, 256>>>(input, col, N, C_in, H, W);
    // dW = sum_n dY[n] * col[n]^T: [C_out, C_in*9]
    for (size_t n = 0; n < N; ++n) {
        const float *col_n = col + n * C_in * 9 * H * W;
        const float *dY_n = grad_output + n * C_out * H * W;
        float beta = (n == 0) ? 0.0f : 1.0f;
        gemm_gpu(false, true, C_out, C_in * 9, H * W, 1.0f, dY_n, col_n, beta, grad_weight);
    }
    cudaFree(col);
#else
    std::vector<float> col(col_size, 0.0f);
    std::fill(grad_weight, grad_weight + C_out * C_in * 9, 0.0f);
    for (size_t n = 0; n < N; ++n) {
        float *col_n = col.data() + n * C_in * 9 * H * W;
        for (size_t c = 0; c < C_in; ++c)
            for (size_t kh = 0; kh < 3; ++kh)
                for (size_t kw = 0; kw < 3; ++kw)
                    for (size_t i = 0; i < H; ++i)
                        for (size_t j = 0; j < W; ++j) {
                            int h_in = i + kh - 1, w_in = j + kw - 1;
                            float val = (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W)
                                ? input[((n * C_in + c) * H + h_in) * W + w_in] : 0.0f;
                            size_t k = c * 9 + kh * 3 + kw;
                            col_n[k * H * W + i * W + j] = val;
                        }
        const float *dY_n = grad_output + n * C_out * H * W;
        float beta = (n == 0) ? 0.0f : 1.0f;
        gemm_gpu(false, true, C_out, C_in * 9, H * W, 1.0f, dY_n, col_n, beta, grad_weight);
    }
#endif
}
