#include <cuda.h>
#include <cuda_fp16.h>

#include "test_util.h"

__global__ void ConvertToHalfKernel(const float* x, half* y, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    y[idx] = __float2half(x[idx]);
  }
}

__global__ void ConvertToFloatKernel(const half* x, float* y, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    y[idx] = __half2float(x[idx]);
  }
}

void ConvertToHalf(cudaStream_t stream, const float* x, half* y, int64_t numel) {
  int block_dim = 256;
  int grid_dim = (numel + 255) / 256;
  ConvertToHalfKernel<<<grid_dim, block_dim, 0, stream>>>(x, y, numel);
  CHECK_CUDA(cudaGetLastError());
}

void ConvertToFloat(cudaStream_t stream, const half* x, float* y, int64_t numel) {
  int block_dim = 256;
  int grid_dim = (numel + 255) / 256;
  ConvertToFloatKernel<<<grid_dim, block_dim, 0, stream>>>(x, y, numel);
  CHECK_CUDA(cudaGetLastError());
}
