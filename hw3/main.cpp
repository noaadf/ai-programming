#include "layers.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ===== 测试辅助 =====

static int pass_count = 0, fail_count = 0;

static void check(const char *name, const float *got, const float *expected, int n,
    float tol = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(got[i] - expected[i]) > tol) {
            printf("FAIL  %s  [%d]: got %.6f, expected %.6f\n", name, i, got[i], expected[i]);
            ++fail_count;
            return;
        }
    }
    printf("PASS  %s\n", name);
    ++pass_count;
}

static void check_scalar(const char *name, float got, float expected, float tol = 1e-4f) {
    if (std::fabs(got - expected) > tol) {
        printf("FAIL  %s: got %.6f, expected %.6f\n", name, got, expected);
        ++fail_count;
    } else {
        printf("PASS  %s\n", name);
        ++pass_count;
    }
}

// ===== FC 层测试 =====
// N=2, C_in=3, C_out=2
// input  = [[1,2,3],[4,5,6]]
// weight = [[1,0,1],[0,1,0]]  (C_out x C_in)
// bias   = [1, 2]
// output = input * weight^T + bias
//        = [[4,2],[10,5]] + [[1,2],[1,2]] = [[5,4],[11,7]]

static void test_fc_forward() {
    float input[]  = {1,2,3, 4,5,6};
    float weight[] = {1,0,1, 0,1,0};
    float bias[]   = {1, 2};
    float output[4] = {};
    float expected[] = {5,4, 11,7};
    fc_forward(input, weight, bias, output, 2, 3, 2);
    check("fc_forward", output, expected, 4);
}

// grad_output = [[1,0],[0,1]]
// grad_input  = grad_output * weight = [[1,0,1],[0,1,0]]
static void test_fc_backward_input() {
    float grad_output[] = {1,0, 0,1};
    float weight[]      = {1,0,1, 0,1,0};
    float grad_input[6] = {};
    float expected[]    = {1,0,1, 0,1,0};
    fc_backward_input(grad_output, weight, grad_input, 2, 3, 2);
    check("fc_backward_input", grad_input, expected, 6);
}

// grad_weight = grad_output^T * input = [[1,2,3],[4,5,6]]
static void test_fc_backward_weight() {
    float grad_output[] = {1,0, 0,1};
    float input[]       = {1,2,3, 4,5,6};
    float grad_weight[6] = {};
    float expected[]    = {1,2,3, 4,5,6};
    fc_backward_weight(grad_output, input, grad_weight, 2, 3, 2);
    check("fc_backward_weight", grad_weight, expected, 6);
}

// grad_output = [[1,2],[3,4]]  → grad_bias = [4, 6]
static void test_fc_backward_bias() {
    float grad_output[] = {1,2, 3,4};
    float grad_bias[2]  = {};
    float expected[]    = {4, 6};
    fc_backward_bias(grad_output, grad_bias, 2, 2);
    check("fc_backward_bias", grad_bias, expected, 2);
}

// ===== Pool 层测试 =====
// N=1, C=1, H=4, W=4
// input (flat):
//  1  3  2  4
//  5  7  6  8
//  9 11 10 12
// 13 15 14 16
// 2x2 max pooling:
//   [0,0]: max(1,3,5,7)=7,   mask=5
//   [0,1]: max(2,4,6,8)=8,   mask=7
//   [1,0]: max(9,11,13,15)=15, mask=13
//   [1,1]: max(10,12,14,16)=16, mask=15

static void test_pool_forward() {
    float input[] = {1,3,2,4, 5,7,6,8, 9,11,10,12, 13,15,14,16};
    float output[4] = {};
    int   mask[4]   = {};
    float exp_out[] = {7, 8, 15, 16};
    int   exp_mask[]= {5, 7, 13, 15};
    pool_forward(input, output, mask, 1, 1, 4, 4);
    check("pool_forward_output", output, exp_out, 4);
    // 检查 mask（转成 float 比较）
    float mask_f[4], exp_mask_f[4];
    for (int i = 0; i < 4; ++i) { mask_f[i] = mask[i]; exp_mask_f[i] = exp_mask[i]; }
    check("pool_forward_mask", mask_f, exp_mask_f, 4);
}

// grad_output = [1,2,3,4]，mask = [5,7,13,15]
// grad_input[5]=1, [7]=2, [13]=3, [15]=4，其余为0
static void test_pool_backward() {
    float grad_output[] = {1, 2, 3, 4};
    int   mask[]        = {5, 7, 13, 15};
    float grad_input[16] = {};
    float expected[16]   = {};
    expected[5] = 1; expected[7] = 2; expected[13] = 3; expected[15] = 4;
    pool_backward(grad_output, mask, grad_input, 1, 1, 4, 4);
    check("pool_backward", grad_input, expected, 16);
}

// ===== Softmax 测试 =====
// N=1, C=3, input=[1,2,3]
// max=3, exp=[e^-2, e^-1, e^0]=[0.13534,0.36788,1.0]
// sum=1.50321
// output=[0.09003, 0.24473, 0.66524]

static void test_softmax() {
    float input[]    = {1.0f, 2.0f, 3.0f};
    float output[3]  = {};
    float expected[] = {0.09003f, 0.24473f, 0.66524f};
    softmax_forward(input, output, 1, 3);
    check("softmax_forward", output, expected, 3);

    // 验证概率和为1
    float sum = output[0] + output[1] + output[2];
    check_scalar("softmax_sum=1", sum, 1.0f);
}

// ===== Cross Entropy 测试 =====
// N=2, C=3
// probs  = [[0.1,0.7,0.2],[0.3,0.3,0.4]]
// labels = [1, 2]
// loss   = -(log(0.7)+log(0.4))/2 = 0.63658

static void test_cross_entropy() {
    float probs[]  = {0.1f,0.7f,0.2f, 0.3f,0.3f,0.4f};
    float labels[] = {1.0f, 2.0f};
    float loss = cross_entropy_forward(probs, labels, 2, 3);
    float expected_loss = -(std::log(0.7f) + std::log(0.4f)) / 2.0f;
    check_scalar("cross_entropy_forward", loss, expected_loss);

    // 反向：grad[n][c] = probs[n][c]/N，label位置再减1/N
    float grad[6] = {};
    float expected_grad[] = {
        0.05f, 0.35f-0.5f, 0.1f,   // n=0, label=1
        0.15f, 0.15f, 0.2f-0.5f    // n=1, label=2
    };
    cross_entropy_backward(probs, labels, grad, 2, 3);
    check("cross_entropy_backward", grad, expected_grad, 6);
}

// ===== Conv 层测试 =====
// N=1, C_in=1, C_out=1, H=3, W=3
// weight = identity kernel: 中心=1，其余=0
// bias = 0
// 期望输出 == 输入

static void test_conv_forward() {
    float input[]  = {1,2,3, 4,5,6, 7,8,9};
    float weight[] = {0,0,0, 0,1,0, 0,0,0};  // identity
    float bias[]   = {0};
    float output[9] = {};
    conv_forward(input, weight, bias, output, 1, 1, 1, 3, 3);
    check("conv_forward_identity", output, input, 9);
}

// 用数值梯度验证 conv_backward_input
// f(x) = sum(conv_forward(x)), df/dx_i ≈ (f(x+eps)-f(x-eps))/(2*eps)
static void test_conv_backward_input() {
    const float eps = 1e-3f;
    float input[]  = {1,2,3, 4,5,6, 7,8,9};
    float weight[] = {1,0,-1, 2,0,-2, 1,0,-1};  // Sobel-x
    float bias[]   = {0};
    float output_p[9] = {}, output_m[9] = {};

    // 解析梯度
    float grad_out[9];  for (int i = 0; i < 9; ++i) grad_out[i] = 1.0f;
    float grad_input[9] = {};
    conv_backward_input(grad_out, weight, grad_input, 1, 1, 1, 3, 3);

    // 数值梯度
    float num_grad[9] = {};
    for (int i = 0; i < 9; ++i) {
        float inp_p[9], inp_m[9];
        memcpy(inp_p, input, sizeof(input)); inp_p[i] += eps;
        memcpy(inp_m, input, sizeof(input)); inp_m[i] -= eps;
        conv_forward(inp_p, weight, bias, output_p, 1, 1, 1, 3, 3);
        conv_forward(inp_m, weight, bias, output_m, 1, 1, 1, 3, 3);
        float sum_p = 0, sum_m = 0;
        for (int j = 0; j < 9; ++j) { sum_p += output_p[j]; sum_m += output_m[j]; }
        num_grad[i] = (sum_p - sum_m) / (2 * eps);
    }
    check("conv_backward_input", grad_input, num_grad, 9, 1e-2f);
}

static void test_conv_backward_weight() {
    const float eps = 1e-3f;
    float input[]  = {1,2,3, 4,5,6, 7,8,9};
    float weight[] = {1,0,-1, 2,0,-2, 1,0,-1};
    float bias[]   = {0};
    float output_p[9] = {}, output_m[9] = {};

    float grad_out[9]; for (int i = 0; i < 9; ++i) grad_out[i] = 1.0f;
    float grad_weight[9] = {};
    conv_backward_weight(grad_out, input, grad_weight, 1, 1, 1, 3, 3);

    float num_grad[9] = {};
    for (int i = 0; i < 9; ++i) {
        float w_p[9], w_m[9];
        memcpy(w_p, weight, sizeof(weight)); w_p[i] += eps;
        memcpy(w_m, weight, sizeof(weight)); w_m[i] -= eps;
        conv_forward(input, w_p, bias, output_p, 1, 1, 1, 3, 3);
        conv_forward(input, w_m, bias, output_m, 1, 1, 1, 3, 3);
        float sum_p = 0, sum_m = 0;
        for (int j = 0; j < 9; ++j) { sum_p += output_p[j]; sum_m += output_m[j]; }
        num_grad[i] = (sum_p - sum_m) / (2 * eps);
    }
    check("conv_backward_weight", grad_weight, num_grad, 9, 1e-2f);
}

// ===== main =====

int main() {
    printf("===== FC =====\n");
    test_fc_forward();
    test_fc_backward_input();
    test_fc_backward_weight();
    test_fc_backward_bias();

    printf("===== Pool =====\n");
    test_pool_forward();
    test_pool_backward();

    printf("===== Softmax =====\n");
    test_softmax();

    printf("===== Cross Entropy =====\n");
    test_cross_entropy();

    printf("===== Conv =====\n");
    test_conv_forward();
    test_conv_backward_input();
    test_conv_backward_weight();

    printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
