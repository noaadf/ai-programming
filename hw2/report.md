# HW2-2 实验报告

## 一、实现概述

本次作业分两部分：实现 `Tensor` 类和激活函数（ReLU、Sigmoid）。

---

## 二、Tensor 类实现

### 数据结构设计

```cpp
enum class Device { CPU, GPU };

class Tensor {
public:
    std::vector<size_t> shape;   // 各维度大小
    Device device;               // 数据所在设备
    std::shared_ptr<float[]> data; // 数据指针（CPU或GPU）
};
```

- 用 `enum class Device` 表示设备类型，避免字符串拼写错误
- 用 `std::vector<size_t>` 存储 shape，支持任意维度
- 用 `std::shared_ptr<float[]>` 管理内存，自动释放，无需手动 `delete`

### 内存分配

构造函数根据设备类型分配不同内存：

```cpp
// CPU：普通堆内存
data = std::shared_ptr<float[]>(new float[total_size]);

// GPU：显存，使用自定义 deleter 确保用 cudaFree 释放
float* gpu_ptr;
cudaMalloc(&gpu_ptr, total_size * sizeof(float));
data = std::shared_ptr<float[]>(gpu_ptr, [](float* p){ cudaFree(p); });
```

### 设备间数据迁移

`.cpu()` 和 `.gpu()` 方法通过 `cudaMemcpy` 在设备间拷贝数据：

- 若已在目标设备，直接返回 `*this`（共享内存，零拷贝）
- 否则分配新内存并拷贝

---

## 三、激活函数实现

### ReLU

**前向传播**：$\text{ReLU}(x) = \max(x, 0)$

**反向传播**：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$ if $x > 0$，else $0$

实现时保存前向输入 `input_`，反向传播时用于判断掩码：

```cpp
// CPU forward
output.data[i] = std::max(0.0f, input.data[i]);

// CPU backward
grad_input.data[i] = input_.data[i] > 0.0f ? grad_output.data[i] : 0.0f;
```

### Sigmoid

**前向传播**：$\sigma(x) = \frac{1}{1 + e^{-x}}$

**反向传播**：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot y(1-y)$

实现时保存前向输出 `output_`，反向传播直接复用：

```cpp
// CPU forward
output.data[i] = 1.0f / (1.0f + expf(-input.data[i]));

// CPU backward
grad_input.data[i] = grad_output.data[i] * output_.data[i] * (1.0f - output_.data[i]);
```

GPU 版本通过 CUDA kernel 并行计算，用 `#ifdef __CUDACC__` 与 CPU 代码隔离。

---

## 四、正确性验证

### 验证方法

编写 `main.cpp` 对 CPU 路径进行数值验证，手动计算期望值与程序输出对比。

**ReLU 测试**：输入 `[-1, 0, 2, -3, 4]`

| 位置 | 输入 | forward 期望 | backward 期望（grad全1） |
|------|------|-------------|------------------------|
| 0    | -1   | 0           | 0                      |
| 1    | 0    | 0           | 0                      |
| 2    | 2    | 2           | 1                      |
| 3    | -3   | 0           | 0                      |
| 4    | 4    | 4           | 1                      |

**Sigmoid 测试**：输入 `[0, 1, -1]`

| 位置 | 输入 | forward 期望 | backward 期望（grad全1） |
|------|------|-------------|------------------------|
| 0    | 0    | 0.5         | 0.25                   |
| 1    | 1    | 0.7311      | 0.1966                 |
| 2    | -1   | 0.2689      | 0.1966                 |

### GPU 验证

在远程服务器（H100）上编译运行，将 GPU 结果通过 `.cpu()` 拷回后与 CPU 结果对比，数值一致则验证通过。

---

## 五、文件结构

```
hw2/
├── tensor.h          # Tensor 类声明
├── tensor.cpp        # Tensor 类实现
├── activation.h      # ReLU / Sigmoid 类声明
└── activation.cpp    # ReLU / Sigmoid 实现（含 CUDA kernel）
```
