#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


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


template <typename T>
T* AllocateAndInit(cudaStream_t stream, size_t numel, bool random, T value = 0, std::vector<float> ref = std::vector<float>{}) {
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
    if (ref.size() == numel) {
      for (size_t i = 0; i < numel; ++i) {
        data[i] = ref[i];
      }
    } else {
      for (size_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(value);
      }
    }
  }

  if constexpr (std::is_same<T, float>::value) {
    std::cout << "-- [AllocateAndInit] dtype=float, numel=" << numel << std::endl;
    CHECK_CUDA(cudaMemcpyAsync(addr, data.data(), numel * sizeof(T), cudaMemcpyHostToDevice, stream));
  } else if constexpr (std::is_same<T, half>::value) {
    std::cout << "-- [AllocateAndInit] dtype=half, numel=" << numel << std::endl;
    float* tmp_addr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), numel * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(tmp_addr, data.data(), numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    ConvertToHalf(stream, tmp_addr, addr, numel);
    cudaFree(tmp_addr);
  } else {
    std::cerr << "-- [AllocateAndInit] Unsupported data type!" << std::endl;
  }

  return addr;
}

template <typename T>
void Print(cudaStream_t stream, T* addr, size_t batch_count, size_t m, size_t n) {
  size_t numel = batch_count * m * n;

  std::vector<float> data;
  data.resize(numel);

  if constexpr (std::is_same<T, float>::value) {
    CHECK_CUDA(cudaMemcpyAsync(data.data(), addr, numel * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  } else if constexpr (std::is_same<T, half>::value) {
    float* tmp_addr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), numel * sizeof(float)));
    ConvertToFloat(stream, addr, tmp_addr, numel);
    CHECK_CUDA(cudaMemcpyAsync(data.data(), tmp_addr, numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaFree(tmp_addr);
  } else {
    std::cerr << "Unsupported!" << std::endl;
  }

  std::cout << std::endl;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      std::cout << data[i * n + j] << ", ";
    }
    std::cout << " ...";
    for (size_t j = n - 5; j < n; ++j) {
      std::cout << ", " << data[i * n + j];
    }
    std::cout << std::endl;
  }
  std::cout << "..." << std::endl;
  for (size_t i = m - 2; i < m; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      std::cout << data[i * n + j] << ", ";
    }
    std::cout << " ...";
    for (size_t j = n - 5; j < n; ++j) {
      std::cout << ", " << data[i * n + j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ConvertToHalf(cudaStream_t stream, const float* x, half* y, int64_t numel);
void ConvertToFloat(cudaStream_t stream, const half* x, float* y, int64_t numel);
