#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <type_traits>

#include "matmul.h"


template <typename T>
T* Allocate(size_t numel, bool set_zero) {
  T* addr = nullptr;
  size_t nbytes = sizeof(T) * numel;

  // Allocate device memory.
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&addr), nbytes));

  if (set_zero) {
    CHECK_CUDA(cudaMemset(addr, 0, numel));
  } else {
    srand((unsigned)time(0));
    std::vector<float> data;
    data.resize(numel);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(5, 2);
    for (size_t i = 0; i < numel; ++i) {
      data[i] = d(gen);
    }
    if (std::is_same<T, float>::value) {
      CHECK_CUDA(cudaMemcpy(addr, data.data(), nbytes, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, half>::value) {
      float* tmp_addr = nullptr;
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), nbytes));
      CHECK_CUDA(cudaMemcpy(tmp_addr, data.data(), nbytes, cudaMemcpyHostToDevice));
      ConvertToHalf(tmp_addr, addr, numel);
      cudaFree(tmp_addr);
    } else {
      std::cerr << "Unsupported!" << std::endl;
    }
  }

  return addr;
}

int ProfileBestConfig(GemmEpilogueParams &params, bool profile) {
  std::cout << "we are tunning for problem: [" << params.m << ", " << params.n
            << ", " << params.k << "]" << std::endl;

  constexpr int kWarmupIters = 10;
  constexpr int kRepeatIters = 10;
  float min_time = 100000.f;
  int min_time_index = -1;

  CHECK_CUDA(
      cudaMemset(params.output, 0, sizeof(half) * params.m * params.n));

  std::cout << "============ Call CUTLASS MatMul ============\n";
  cutlass::Status status = CutlassMatmulAddFp16(params);
  std::cout << "=============================================\n";
  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  if (profile) {
    for (int i = 0; i < kWarmupIters; i++) {
      status = CutlassMatmulAddFp16(params);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventCreate(&beg));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(beg));
    for (int i = 0; i < kRepeatIters; i++) {
      status = CutlassMatmulAddFp16(params);
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

int main(int argc, const char *arg[]) {
  GemmEpilogueParams params;
  params.m = 256;
  params.n = 256;
  params.k = 256;
  params.lda = params.k;
  params.ldb = params.n;
  params.ldd = params.n;

  params.input = Allocate<half>(params.m * params.k, false);
  params.weight = Allocate<half>(params.k * params.n, false);
  params.bias = Allocate<half>(params.n, false);
  params.output = Allocate<half>(params.m * params.n, true);

  ProfileBestConfig(params, false);

  cudaFree(params.input);
  cudaFree(params.weight);
  cudaFree(params.bias);
  cudaFree(params.output);

  return 0;
}

