/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/* Using updated (v2) interfaces to cublas */
#include </softs/nvidia/cuda-10.1/include/cuda.h>
#include </softs/nvidia/cuda-10.1/include/cuda_runtime.h>
#include </softs/nvidia/cuda-10.1/include/cusparse.h>
#include </softs/nvidia/cuda-10.1/include/cublas_v2.h>

// Utilities and system includes
#include </softs/nvidia/cuda-10.1/samples/common/inc/helper_functions.h>  // helper for shared functions common to CUDA Samples
#include </softs/nvidia/cuda-10.1/samples/common/inc/helper_cuda.h>       // helper function CUDA error checking and initialization

#include "CG.h"
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <torch/extension.h>

namespace plasma {

typedef at::Tensor T;

const char *sSDKname     = "conjugateGradient";


//int Conjugate_Gradient
void Conjugate_Gradient
(
 T div_vec,
 T A_val,
 T I_A,
 T J_A,
 const float tol,
 const int max_iter,
 int M,
 int N,
 int nz,
 T p,
 float &r1
){
    int *I = NULL, *J = NULL;
    float *val = NULL;
    float *x;

    float *rhs= NULL;
    float a, b, na, r0;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    // T p = zeros_like(flags);
    T residual = zeros_like(p);
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;

    I = I_A.data<int>();
    J = J_A.data<int>();
    val = A_val.data<float>();
    rhs = div_vec.data<float>();
    x = p.data<float>();
    //std::cout << "nz  -------------------- "<< nz << std::endl;

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);


    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));


    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);


    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    //cusparseCsrmvEx(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);


    k = 1;

    //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        //cusparseCsrmvEx(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    
   
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

}



void solveLinearSystemCG
(
 T pot,
 T div_vec,
 T A_val,
 T I_A,
 T J_A,
 const float p_tol,
 const int max_iter,
 float &residue
 ) {

      // Check arguments.
      int h = pot.size(0);
      int w = pot.size(1);
      int nnz = A_val.size(0);


      // CG TEST
      std::cout << "BEGIN CG TEST  -------------------- "<< std::endl;
      Conjugate_Gradient(div_vec,A_val, I_A, J_A, p_tol, max_iter, h*w,h*w,nnz, pot, residue);
      std::cout << "END CG TEST  -------------------- "<< std::endl;
}

} // Fluid namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_linear_system_CG", &(plasma::solveLinearSystemCG), "Solve Linear System using CG method");
}
