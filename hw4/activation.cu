#include "activation.h"
#include <cmath>
#include <stdexcept>

#ifdef __CUDACC__
__global__ void relu_forward_kernel(float* out, const float* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void relu_backward_kernel(float* grad_in, const float* grad_out, const float* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = in[i] > 0.0f ? grad_out[i] : 0.0f;
}
__global__ void sigmoid_forward_kernel(float* out, const float* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void sigmoid_backward_kernel(float* grad_in, const float* grad_out, const float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = grad_out[i] * out[i] * (1.0f - out[i]);
}

#endif

// ========== ReLU ==========

Tensor ReLU::forward(const Tensor& input) {
    input_ = input;
    Tensor output(input.shape, input.device);
    if (input.device == Device::CPU) {
        for (size_t i = 0; i < input.size(); ++i) {
            output.data[i] = std::max(0.0f, input.data[i]);
        }
    } else {
#ifdef __CUDACC__
        size_t n = input.size();
        relu_forward_kernel<<<(n + 255) / 256, 256>>>(output.data.get(), input.data.get(), n);
#else
        throw std::runtime_error("GPU support is not available.");
#endif
    }
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output.shape, grad_output.device);
    if (grad_output.device == Device::CPU) {
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input.data[i] = input_.data[i] > 0.0f ? grad_output.data[i] : 0.0f;
        }
    } else {
#ifdef __CUDACC__
        size_t n = grad_output.size();
        relu_backward_kernel<<<(n + 255) / 256, 256>>>(grad_input.data.get(), grad_output.data.get(), input_.data.get(), n);
#else
        throw std::runtime_error("GPU support is not available.");
#endif
    }
    return grad_input;
}

// ========== Sigmoid ==========

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output(input.shape, input.device);
    if (input.device == Device::CPU) {
        for (size_t i = 0; i < input.size(); ++i) {
            output.data[i] = 1.0f / (1.0f + expf(-input.data[i]));
        }
    } else {
#ifdef __CUDACC__
        size_t n = input.size();
        sigmoid_forward_kernel<<<(n + 255) / 256, 256>>>(output.data.get(), input.data.get(), n);
#else
        throw std::runtime_error("GPU support is not available.");
#endif
    }
    output_ = output;
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    Tensor grad_input(grad_output.shape, grad_output.device);
    if (grad_output.device == Device::CPU) {
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input.data[i] = grad_output.data[i] * output_.data[i] * (1.0f - output_.data[i]);
        }
    } else {
#ifdef __CUDACC__
        size_t n = grad_output.size();
        sigmoid_backward_kernel<<<(n + 255) / 256, 256>>>(grad_input.data.get(), grad_output.data.get(), output_.data.get(), n);
#else
        throw std::runtime_error("GPU support is not available.");
#endif
    }
    return grad_input;
}
