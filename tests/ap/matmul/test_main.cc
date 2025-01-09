#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <type_traits>

#include "matmul.h"


template <typename T>
T* AllocateAndInit(size_t numel, bool random, T value = 0) {
  T* addr = nullptr;

  // Allocate device memory.
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&addr), numel * sizeof(T)));

  std::vector<float> data;
  data.resize(numel);

  if (random) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(5, 2);
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<float>(d(gen));
    }
  } else {
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<float>(value);
    }
  }

  if (std::is_same<T, float>::value) {
    std::cout << "-- AllocateAndInit: dtype=float, numel=" << numel << std::endl;
    CHECK_CUDA(cudaMemcpy(addr, data.data(), numel * sizeof(T), cudaMemcpyHostToDevice));
  } else if (std::is_same<T, half>::value) {
    std::cout << "-- AllocateAndInit: dtype=half, numel=" << numel << std::endl;
    float* tmp_addr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), numel * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(tmp_addr, data.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    ConvertToHalf(tmp_addr, addr, numel);
    cudaFree(tmp_addr);
  } else {
    std::cerr << "Unsupported!" << std::endl;
  }

  return addr;
}

template <typename T>
void Print(T* addr, size_t numel) {
  std::vector<float> data;
  data.resize(numel);

  if (std::is_same<T, float>::value) {
    CHECK_CUDA(cudaMemcpy(data.data(), addr, numel * sizeof(T), cudaMemcpyDeviceToHost));
  } else if (std::is_same<T, half>::value) {
    float* tmp_addr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), numel * sizeof(float)));
    ConvertToFloat(addr, tmp_addr, numel);
    CHECK_CUDA(cudaMemcpy(data.data(), tmp_addr, numel * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(tmp_addr);
  } else {
    std::cerr << "Unsupported!" << std::endl;
  }

  std::cout << std::endl;
  for (size_t i = 0; i < 10; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << data[i];
  }
  std::cout << std::endl;
}

template <typename T>
void RunAndCheckAccuracy(GemmEpilogueParams &params) {
  std::cout << "we are tunning for problem: [" << params.m << ", " << params.n
            << ", " << params.k << "]" << std::endl;

  CHECK_CUDA(
      cudaMemset(params.output, 0, sizeof(T) * params.m * params.n));
  CHECK_CUTLASS(CutlassMatmulAddUnary(params));

  Print<T>(reinterpret_cast<T*>(params.output), params.m * params.n);
}

int main(int argc, const char *arg[]) {
  GemmEpilogueParams params;
  params.m = 256;
  params.n = 256;
  params.k = 256;
  params.lda = params.k;
  params.ldb = params.n;
  params.ldd = params.n;

  using DType = half;

  params.input = AllocateAndInit<DType>(params.m * params.k, false, 1.);
  params.weight = AllocateAndInit<DType>(params.k * params.n, false, 1.);
  params.bias = AllocateAndInit<DType>(params.n, false, 1000.);
  params.output = AllocateAndInit<DType>(params.m * params.n, false, 0.);

  RunAndCheckAccuracy<DType>(params);

  cudaFree(params.input);
  cudaFree(params.weight);
  cudaFree(params.bias);
  cudaFree(params.output);

  return 0;
}
