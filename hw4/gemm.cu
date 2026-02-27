#include "gemm.h"
#include <cstring>

#ifdef __CUDACC__
#include <cublas_v2.h>
#endif

// GPU 实现：封装 cublasSgemm，对外暴露行主序接口
// 技巧：cublasSgemm 是列主序，行主序的 C=op(A)*op(B)
//       等价于列主序的 C^T = op(B)^T * op(A)^T
//       所以交换 A/B 的位置和转置标志即可
void gemm_gpu(bool trans_a, bool trans_b,
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C) {
#ifdef __CUDACC__
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t op_A = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = trans_a ? M : K;  // A 的 leading dimension
    int ldb = trans_b ? K : N;  // B 的 leading dimension

    // 交换 A/B 顺序，同时交换 M/N，完成行列主序转换
    cublasSgemm(handle, op_B, op_A, N, M, K,
                &alpha, B, ldb, A, lda, &beta, C, N);

    cublasDestroy(handle);
#else
    // 没有 CUDA 时退回 CPU 实现
    gemm_cpu(trans_a, trans_b, M, N, K, alpha, A, B, beta, C);
#endif
}

// CPU 实现：三重循环，支持转置
void gemm_cpu(bool trans_a, bool trans_b,
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = trans_a ? A[k * M + m] : A[m * K + k];
                float b = trans_b ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}
