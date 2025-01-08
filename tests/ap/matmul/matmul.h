#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CHECK_CUDA(func)                                                            \
  {                                                                                 \
    cudaError_t err = func;                                                         \
    if (err != cudaSuccess) {                                                       \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", "                       \
                << __FUNCTION__ << "] " << "CUDA error(" << err << "), "            \
                << cudaGetErrorString(err) << " when call " << #func << std::endl;  \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }


void ConvertToHalf(const float* x, half* y, int64_t numel);
void ConvertToFloat(const half* x, float* y, int64_t numel);


struct GemmEpilogueParams {
  void *input;
  void *weight;
  void *bias;
  void *output;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldd;
  cudaStream_t stream;
  bool is_vec_bias = true;
  int sm_version = 80;
  void *workspace = nullptr;
};

void NativeMatmulAdd(const GemmEpilogueParams& params);
cutlass::Status CutlassMatmulAdd(const GemmEpilogueParams& params);
