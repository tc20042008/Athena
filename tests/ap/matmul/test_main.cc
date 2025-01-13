#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <type_traits>

#include "kernel.h"
#include "util.h"


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

  if (std::is_same<T, float>::value) {
    std::cout << "-- AllocateAndInit: dtype=float, numel=" << numel << std::endl;
    CHECK_CUDA(cudaMemcpyAsync(addr, data.data(), numel * sizeof(T), cudaMemcpyHostToDevice, stream));
  } else if (std::is_same<T, half>::value) {
    std::cout << "-- AllocateAndInit: dtype=half, numel=" << numel << std::endl;
    float* tmp_addr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tmp_addr), numel * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(tmp_addr, data.data(), numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    ConvertToHalf(stream, tmp_addr, addr, numel);
    cudaFree(tmp_addr);
  } else {
    std::cerr << "Unsupported!" << std::endl;
  }

  return addr;
}

template <typename T>
void Print(cudaStream_t stream, T* addr, size_t m, size_t n) {
  size_t numel = m * n;

  std::vector<float> data;
  data.resize(numel);

  if (std::is_same<T, float>::value) {
    CHECK_CUDA(cudaMemcpyAsync(data.data(), addr, numel * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  } else if (std::is_same<T, half>::value) {
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

template <typename T>
void TestMatmulAddUnary(cudaStream_t stream) {
  int m = 256;
  int n = 512;
  int k = 256;

  std::cout << "we are running for problem: [" << m << ", " << n
            << ", " << k << "]" << std::endl;

  T* input = AllocateAndInit<T>(stream, m * k, false, 1.);
  T* weight = AllocateAndInit<T>(stream, k * n, false, 1.);

  std::vector<float> bias_ref;
  bias_ref.resize(n);
  for (size_t i = 0; i < bias_ref.size(); ++i) {
    bias_ref[i] = static_cast<float>(1000 * (i % 10));
  }
  T* bias = AllocateAndInit<T>(stream, n, false, 0., bias_ref);

  T* output = AllocateAndInit<T>(stream, m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T) * m * n, stream));
  MatmulAddUnaryKernel(stream, input, weight, bias, output, m, n, k);

  Print<T>(stream, reinterpret_cast<T*>(output), m, n);

  cudaFree(input);
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(output);
}

// template <typename T>
// void TestMatmulAddBinary(cudaStream_t stream) {
//   GemmBroadcastEpilogueParams params;
//   params.m = 256;
//   params.n = 512;
//   params.k = 256;
// 
//   std::cout << "we are tunning for problem: [" << params.m << ", " << params.n
//             << ", " << params.k << "]" << std::endl;
// 
//   params.input = AllocateAndInit<T>(params.m * params.k, false, 1.);
//   params.weight = AllocateAndInit<T>(params.k * params.n, false, 1.);
// 
//   std::vector<float> bias_ref;
//   bias_ref.resize(params.n);
//   for (size_t i = 0; i < bias_ref.size(); ++i) {
//     bias_ref[i] = static_cast<float>(1000 * (i % 10));
//   }
//   params.bias = AllocateAndInit<T>(params.n, false, 0., bias_ref);
// 
//   std::vector<float> broadcast_ref;
//   broadcast_ref.resize(params.m);
//   for (size_t i = 0; i < broadcast_ref.size(); ++i) {
//     broadcast_ref[i] = static_cast<float>(10000 * (i % 5));
//   }
//   params.broadcast = AllocateAndInit<T>(params.m, false, 0., broadcast_ref);
// 
//   params.output = AllocateAndInit<T>(params.m * params.n, false, 0.);
//   params.broadcast_out = AllocateAndInit<T>(params.m * params.n, false, 0.);
// 
//   CHECK_CUDA(
//       cudaMemsetAsync(params.output, 0, sizeof(T) * params.m * params.n, stream));
//   CHECK_CUDA(
//       cudaMemsetAsync(params.broadcast_out, 0, sizeof(T) * params.m * params.n, stream));
//   CHECK_CUTLASS(CutlassMatmulAddBinary(params));
// 
//   Print<T>(reinterpret_cast<T*>(params.output), params.m, params.n);
// 
//   cudaFree(params.input);
//   cudaFree(params.weight);
//   cudaFree(params.bias);
//   cudaFree(params.output);
//   cudaFree(params.broadcast);
//   cudaFree(params.broadcast_out);
// }

int main(int argc, const char *arg[]) {
  using DType = half;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  TestMatmulAddUnary<DType>(stream);
  // TestMatmulAddBinary<DType>(stream);

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
