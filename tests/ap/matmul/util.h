#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>


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


void ConvertToHalf(cudaStream_t stream, const float* x, half* y, int64_t numel);
void ConvertToFloat(cudaStream_t stream, const half* x, float* y, int64_t numel);
