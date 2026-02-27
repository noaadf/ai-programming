#include <iostream>
#include <cmath>
#include "tensor.h"
#include "activation.h"

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

static void check(const char* name, float got, float expected) {
    bool ok = approx(got, expected);
    std::cout << (ok ? "[PASS] " : "[FAIL] ")
              << name << ": got " << got << ", expected " << expected << "\n";
}

void test_relu() {
    std::cout << "\n=== ReLU ===\n";

    // 输入: [-1, 0, 2, -3, 4]
    Tensor input({5}, Device::CPU);
    input.data[0] = -1.0f;
    input.data[1] =  0.0f;
    input.data[2] =  2.0f;
    input.data[3] = -3.0f;
    input.data[4] =  4.0f;

    ReLU relu;
    Tensor output = relu.forward(input);

    std::cout << "-- forward --\n";
    check("output[0]", output.data[0], 0.0f);
    check("output[1]", output.data[1], 0.0f);
    check("output[2]", output.data[2], 2.0f);
    check("output[3]", output.data[3], 0.0f);
    check("output[4]", output.data[4], 4.0f);

    // grad_output 全为 1
    Tensor grad_out({5}, Device::CPU);
    for (size_t i = 0; i < 5; ++i) grad_out.data[i] = 1.0f;

    Tensor grad_in = relu.backward(grad_out);

    std::cout << "-- backward --\n";
    check("grad[0]", grad_in.data[0], 0.0f);
    check("grad[1]", grad_in.data[1], 0.0f);
    check("grad[2]", grad_in.data[2], 1.0f);
    check("grad[3]", grad_in.data[3], 0.0f);
    check("grad[4]", grad_in.data[4], 1.0f);
}

void test_sigmoid() {
    std::cout << "\n=== Sigmoid ===\n";

    // 输入: [0, 1, -1]
    Tensor input({3}, Device::CPU);
    input.data[0] =  0.0f;
    input.data[1] =  1.0f;
    input.data[2] = -1.0f;

    Sigmoid sigmoid;
    Tensor output = sigmoid.forward(input);

    std::cout << "-- forward --\n";
    check("output[0]", output.data[0], 0.5000f);
    check("output[1]", output.data[1], 0.7311f);
    check("output[2]", output.data[2], 0.2689f);

    // grad_output 全为 1
    Tensor grad_out({3}, Device::CPU);
    for (size_t i = 0; i < 3; ++i) grad_out.data[i] = 1.0f;

    Tensor grad_in = sigmoid.backward(grad_out);

    std::cout << "-- backward --\n";
    check("grad[0]", grad_in.data[0], 0.2500f);
    check("grad[1]", grad_in.data[1], 0.1966f);
    check("grad[2]", grad_in.data[2], 0.1966f);
}

int main() {
    test_relu();
    test_sigmoid();
    return 0;
}
