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

#if 0
int ProfileBestConfig(GemmEpilogueParams &params) {
  std::cout << "we are tunning for problem: [" << params.m << ", " << params.n
            << ", " << params.k << "]" << std::endl;

  constexpr int kWarmupIters = 10;
  constexpr int kRepeatIters = 10;
  float min_time = 100000.f;
  int min_time_index = -1;

  CHECK_CUDA(
      cudaMemset(params.output, 0, sizeof(half) * params.m * params.n));

  std::cout << "============ Call CUTLASS MatMul ============\n";
  cutlass::Status status = CutlassMatmulAdd(params);
  std::cout << "=============================================\n";
  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  if (profile) {
    for (int i = 0; i < kWarmupIters; i++) {
      status = CutlassMatmulAdd(params);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventCreate(&beg));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(beg));
    for (int i = 0; i < kRepeatIters; i++) {
      status = CutlassMatmulAdd(params);
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, beg, end));

    // if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
    //   min_time = elapsed_time;
    //   min_time_index = i;

    //   if (params.data_type == GemmEpilogueDataType::fp16) {
    //     // debug code
    //     std::cout << "fp16_" << OpType2String(op_type) << ": tactic " << i
    //               << " has max diff "
    //               << gemm_epilogue_diff_gpu<half>(params, op_type)
    //               << " compared with baseline,"
    //               << "cost_time: " << elapsed_time << "ms." << std::endl;
    //   } else if (params.data_type == GemmEpilogueDataType::bf16) {
    //     // debug code
    //     std::cout << "bf16_" << OpType2String(op_type) << ": tactic " << i
    //               << " has max diff "
    //               << gemm_epilogue_diff_gpu<__nv_bfloat16>(params, op_type)
    //               << " compared with baseline,"
    //               << "cost_time: " << elapsed_time << "ms." << std::endl;
    //   } else if (params.data_type == GemmEpilogueDataType::fp32) {
    //     // debug code
    //     std::cout << "fp32_" << OpType2String(op_type) << ": tactic " << i
    //               << " has max diff "
    //               << gemm_epilogue_diff_gpu<float>(params, op_type)
    //               << " compared with baseline,"
    //               << "cost_time: " << elapsed_time << "ms." << std::endl;
    //   }
    // }
  }

  return 0;
}
#endif
