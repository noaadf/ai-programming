#pragma once
#include <cstddef>

// ========== gemm_gpu 封装 ==========
// 行主序接口：C = alpha * op(A) * op(B) + beta * C
// trans_a/trans_b: true 表示转置
// A: [M, K] 或 [K, M]（转置时）
// B: [K, N] 或 [N, K]（转置时）
// C: [M, N]
void gemm_gpu(bool trans_a, bool trans_b,
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C);

// CPU fallback（用于本地测试）
void gemm_cpu(bool trans_a, bool trans_b,
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C);
