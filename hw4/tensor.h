#pragma once
#include <vector>
#include <memory>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

enum class Device { CPU, GPU };

class Tensor {
public:

    Tensor() : shape({}), device(Device::CPU), data(nullptr) {}
    Tensor(std::vector<size_t> shape, Device device);
    Tensor cpu() const;
    Tensor gpu() const;
    size_t size() const;

    std::vector<size_t> shape;
    Device device;
    std::shared_ptr<float[]> data;
};

// GPU int 缓冲区辅助（供 Pooling mask 使用）
#ifdef USE_CUDA
int* gpu_alloc_int(size_t n);
void gpu_free_int(int* p);
void gpu_copy_int_h2d(int* dst, const int* src, size_t n);
void gpu_copy_int_d2h(int* dst, const int* src, size_t n);
#endif
