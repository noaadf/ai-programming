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
