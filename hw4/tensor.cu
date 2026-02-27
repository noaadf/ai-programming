#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(std::vector<size_t> shape, Device device) : shape(shape), device(device) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    if (device == Device::CPU) {
        data = std::shared_ptr<float[]>(new float[total_size]);
    } else {
        #ifdef __CUDACC__
                float* gpu_ptr;
                cudaMalloc(&gpu_ptr, total_size * sizeof(float));
                data = std::shared_ptr<float[]>(gpu_ptr, [](float* p){ cudaFree(p); });
        #else
                throw std::runtime_error("GPU support is not available.");
        #endif
    }
}

size_t Tensor::size() const {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    return total_size;
}

Tensor Tensor::cpu() const {
    if (device == Device::CPU) {
        return *this;
    } else {
        Tensor cpu_tensor(shape, Device::CPU);
        #ifdef __CUDACC__
                cudaMemcpy(cpu_tensor.data.get(), data.get(), size() * sizeof(float), cudaMemcpyDeviceToHost);
        #else
                throw std::runtime_error("GPU support is not available.");
        #endif
        return cpu_tensor;
    }
}

Tensor Tensor::gpu() const {
    if (device == Device::GPU) {
        return *this;
    } else {
        Tensor gpu_tensor(shape, Device::GPU);
        #ifdef __CUDACC__
                cudaMemcpy(gpu_tensor.data.get(), data.get(), size() * sizeof(float), cudaMemcpyHostToDevice);
        #else
                throw std::runtime_error("GPU support is not available.");
        #endif
        return gpu_tensor;
    }
}

#ifdef __CUDACC__
int* gpu_alloc_int(size_t n) {
    int* p;
    cudaMalloc(&p, n * sizeof(int));
    return p;
}
void gpu_free_int(int* p) { cudaFree(p); }
void gpu_copy_int_h2d(int* dst, const int* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(int), cudaMemcpyHostToDevice);
}
void gpu_copy_int_d2h(int* dst, const int* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(int), cudaMemcpyDeviceToHost);
}
#endif