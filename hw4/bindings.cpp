#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "activation.h"
#include "layers.h"

namespace py = pybind11;

// ===== GPU 辅助：CUDA 构建时数据自动搬到 GPU =====
#ifdef USE_CUDA
static Tensor to_dev(const Tensor& t) { return t.gpu(); }
static Tensor to_cpu(const Tensor& t) { return t.cpu(); }
#else
static Tensor to_dev(const Tensor& t) { return t; }
static Tensor to_cpu(const Tensor& t) { return t; }
#endif

// ===== Wrapper 类：把自由函数封装为有状态的模块 =====

class FC {
public:
    Tensor weight, bias;
    Tensor input_;  // 缓存，反向传播用
    size_t c_in, c_out;

    FC(size_t c_in, size_t c_out) :
        weight({c_out, c_in}, Device::CPU),
        bias({c_out}, Device::CPU),
        c_in(c_in), c_out(c_out) {}

    Tensor forward(const Tensor& input) {
        input_ = input;
        size_t N = input.shape[0];
        Tensor output({N, c_out}, Device::CPU);
        auto d_in = to_dev(input);
        auto d_w = to_dev(weight);
        auto d_b = to_dev(bias);
        auto d_out = to_dev(output);
        fc_forward(d_in.data.get(), d_w.data.get(), d_b.data.get(),
                   d_out.data.get(), N, c_in, c_out);
        return to_cpu(d_out);
    }

    Tensor backward(const Tensor& grad_output) {
        size_t N = grad_output.shape[0];
        Tensor grad_input({N, c_in}, Device::CPU);
        auto d_go = to_dev(grad_output);
        auto d_w = to_dev(weight);
        auto d_gi = to_dev(grad_input);
        fc_backward_input(d_go.data.get(), d_w.data.get(),
                          d_gi.data.get(), N, c_in, c_out);
        return to_cpu(d_gi);
    }

    Tensor backward_weight(const Tensor& grad_output) {
        size_t N = grad_output.shape[0];
        Tensor grad_w({c_out, c_in}, Device::CPU);
        auto d_go = to_dev(grad_output);
        auto d_in = to_dev(input_);
        auto d_gw = to_dev(grad_w);
        fc_backward_weight(d_go.data.get(), d_in.data.get(),
                           d_gw.data.get(), N, c_in, c_out);
        return to_cpu(d_gw);
    }

    Tensor backward_bias(const Tensor& grad_output) {
        size_t N = grad_output.shape[0];
        Tensor grad_b({c_out}, Device::CPU);
        auto d_go = to_dev(grad_output);
        auto d_gb = to_dev(grad_b);
        fc_backward_bias(d_go.data.get(), d_gb.data.get(), N, c_out);
        return to_cpu(d_gb);
    }
};

class Conv {
public:
    Tensor weight, bias;
    Tensor input_;
    size_t c_in, c_out;

    Conv(size_t c_in, size_t c_out) :
        weight({c_out, c_in, 3, 3}, Device::CPU),
        bias({c_out}, Device::CPU),
        c_in(c_in), c_out(c_out) {}

    Tensor forward(const Tensor& input) {
        input_ = input;
        size_t N = input.shape[0], H = input.shape[2], W = input.shape[3];
        Tensor output({N, c_out, H, W}, Device::CPU);
        auto d_in = to_dev(input);
        auto d_w = to_dev(weight);
        auto d_b = to_dev(bias);
        auto d_out = to_dev(output);
        conv_forward(d_in.data.get(), d_w.data.get(), d_b.data.get(),
                     d_out.data.get(), N, c_in, c_out, H, W);
        return to_cpu(d_out);
    }

    Tensor backward(const Tensor& grad_output) {
        size_t N = grad_output.shape[0], H = input_.shape[2], W = input_.shape[3];
        Tensor grad_input({N, c_in, H, W}, Device::CPU);
        auto d_go = to_dev(grad_output);
        auto d_w = to_dev(weight);
        auto d_gi = to_dev(grad_input);
        conv_backward_input(d_go.data.get(), d_w.data.get(),
                            d_gi.data.get(), N, c_in, c_out, H, W);
        return to_cpu(d_gi);
    }

    Tensor backward_weight(const Tensor& grad_output) {
        size_t N = grad_output.shape[0], H = input_.shape[2], W = input_.shape[3];
        Tensor grad_w({c_out, c_in, 3, 3}, Device::CPU);
        auto d_go = to_dev(grad_output);
        auto d_in = to_dev(input_);
        auto d_gw = to_dev(grad_w);
        conv_backward_weight(d_go.data.get(), d_in.data.get(),
                             d_gw.data.get(), N, c_in, c_out, H, W);
        return to_cpu(d_gw);
    }
};

class Pooling {
public:
    std::vector<int> mask_;
    Tensor input_;
#ifdef USE_CUDA
    int* d_mask_ = nullptr;
    size_t mask_size_ = 0;
#endif

    Tensor forward(const Tensor& input) {
        input_ = input;
        size_t N = input.shape[0], C = input.shape[1];
        size_t H = input.shape[2], W = input.shape[3];
        size_t out_size = N * C * (H / 2) * (W / 2);
        Tensor output({N, C, H / 2, W / 2}, Device::CPU);
        mask_.resize(out_size);
#ifdef USE_CUDA
        if (d_mask_) gpu_free_int(d_mask_);
        d_mask_ = gpu_alloc_int(out_size);
        mask_size_ = out_size;
        auto d_in = to_dev(input);
        auto d_out = to_dev(output);
        pool_forward(d_in.data.get(), d_out.data.get(),
                     d_mask_, N, C, H, W);
        return to_cpu(d_out);
#else
        pool_forward(input.data.get(), output.data.get(),
                     mask_.data(), N, C, H, W);
        return output;
#endif
    }

    Tensor backward(const Tensor& grad_output) {
        size_t N = input_.shape[0], C = input_.shape[1];
        size_t H = input_.shape[2], W = input_.shape[3];
        Tensor grad_input({N, C, H, W}, Device::CPU);
#ifdef USE_CUDA
        auto d_go = to_dev(grad_output);
        auto d_gi = to_dev(grad_input);
        pool_backward(d_go.data.get(), d_mask_,
                      d_gi.data.get(), N, C, H, W);
        return to_cpu(d_gi);
#else
        pool_backward(grad_output.data.get(), mask_.data(),
                      grad_input.data.get(), N, C, H, W);
        return grad_input;
#endif
    }
};

class SoftMax {
public:
    Tensor forward(const Tensor& input) {
        size_t N = input.shape[0], C = input.shape[1];
        Tensor output({N, C}, Device::CPU);
        auto d_in = to_dev(input);
        auto d_out = to_dev(output);
        softmax_forward(d_in.data.get(), d_out.data.get(), N, C);
        return to_cpu(d_out);
    }
};

class CrossEntropyLoss {
public:
    Tensor probs_;  // 缓存 softmax 输出，反向传播用

    float forward(const Tensor& probs, const Tensor& labels) {
        probs_ = probs;
        size_t N = probs.shape[0], C = probs.shape[1];
        auto d_p = to_dev(probs);
        auto d_l = to_dev(labels);
        return cross_entropy_forward(d_p.data.get(), d_l.data.get(), N, C);
    }

    Tensor backward(const Tensor& labels) {
        size_t N = probs_.shape[0], C = probs_.shape[1];
        Tensor grad({N, C}, Device::CPU);
        auto d_p = to_dev(probs_);
        auto d_l = to_dev(labels);
        auto d_g = to_dev(grad);
        cross_entropy_backward(d_p.data.get(), d_l.data.get(),
                               d_g.data.get(), N, C);
        return to_cpu(d_g);
    }
};

// numpy array → Tensor：拷贝数据到我们自己的内存
Tensor tensor_from_numpy(py::array_t<float> arr) {
    // 确保内存连续（C order）
    auto buf = arr.request();
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    Tensor t(shape, Device::CPU);
    std::memcpy(t.data.get(), buf.ptr, t.size() * sizeof(float));
    return t;
}

// Tensor → numpy array：拷贝数据出来给 numpy
py::array_t<float> tensor_to_numpy(const Tensor& t) {
    Tensor cpu_t = t.cpu();  // 确保数据在 CPU 上
    // 把 shape 从 size_t 转成 py::ssize_t
    std::vector<py::ssize_t> shape(cpu_t.shape.begin(), cpu_t.shape.end());
    auto result = py::array_t<float>(shape);
    auto buf = result.request();
    std::memcpy(buf.ptr, cpu_t.data.get(), cpu_t.size() * sizeof(float));
    return result;
}

PYBIND11_MODULE(mytensor, m) {
    m.doc() = "自定义深度学习框架 Python 绑定";

    // ===== Device 枚举 =====
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    // ===== Tensor 类 =====
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>, Device>(),
             py::arg("shape"), py::arg("device") = Device::CPU)
        .def("size", &Tensor::size)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("numpy", &tensor_to_numpy)
        .def_readwrite("shape", &Tensor::shape)
        .def_readonly("device", &Tensor::device);

    // Tensor 工厂函数
    m.def("from_numpy", &tensor_from_numpy, "从 numpy 数组创建 Tensor");

    // ===== ReLU =====
    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward);

    // ===== Sigmoid =====
    py::class_<Sigmoid>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward);

    // ===== FC 全连接层 =====
    py::class_<FC>(m, "FC")
        .def(py::init<size_t, size_t>(), py::arg("c_in"), py::arg("c_out"))
        .def("forward", &FC::forward)
        .def("backward", &FC::backward)
        .def("backward_weight", &FC::backward_weight)
        .def("backward_bias", &FC::backward_bias)
        .def_readwrite("weight", &FC::weight)
        .def_readwrite("bias", &FC::bias);

    // ===== Conv 卷积层 =====
    py::class_<Conv>(m, "Conv")
        .def(py::init<size_t, size_t>(), py::arg("c_in"), py::arg("c_out"))
        .def("forward", &Conv::forward)
        .def("backward", &Conv::backward)
        .def("backward_weight", &Conv::backward_weight)
        .def_readwrite("weight", &Conv::weight)
        .def_readwrite("bias", &Conv::bias);

    // ===== Pooling 池化层 =====
    py::class_<Pooling>(m, "Pooling")
        .def(py::init<>())
        .def("forward", &Pooling::forward)
        .def("backward", &Pooling::backward);

    // ===== SoftMax =====
    py::class_<SoftMax>(m, "SoftMax")
        .def(py::init<>())
        .def("forward", &SoftMax::forward);

    // ===== CrossEntropyLoss =====
    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);
}