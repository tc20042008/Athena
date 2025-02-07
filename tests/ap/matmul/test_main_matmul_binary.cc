#include <iostream>

#include "profile.h"
#include "test_util.h"
#include "kernel.h"


template <typename T>
void TestMatmulAddBinary(cudaStream_t stream) {
  int batch_count = 1;
  int m = 256;
  int n = 512;
  int k = 256;
  bool need_broadcast = false;

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

  int64_t broadcast_numel = need_broadcast ? m : m * n;
  std::vector<float> broadcast_ref;
  broadcast_ref.resize(broadcast_numel);
  if (need_broadcast) {
    for (size_t i = 0; i < broadcast_ref.size(); ++i) {
      broadcast_ref[i] = static_cast<float>(10000 * (i % 5));
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        broadcast_ref[i * n + j] = static_cast<float>(10000 * (i % 5));
      }
    }
  }
  T* broadcast = AllocateAndInit<T>(stream, broadcast_numel, false, 0., broadcast_ref);

  T* output = AllocateAndInit<T>(stream, m * n, false, 0.);
  T* broadcast_out = AllocateAndInit<T>(stream, m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(
      cudaMemsetAsync(output, 0, sizeof(T) * m * n, stream));
  CHECK_CUDA(
      cudaMemsetAsync(broadcast_out, 0, sizeof(T) * m * n, stream));
  MatmulAddBinaryKernel(&stream, input, weight, bias, broadcast, broadcast_out, output, m, n, k, need_broadcast);

  Print<T>(stream, reinterpret_cast<T*>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(output);
  cudaFree(broadcast);
  cudaFree(broadcast_out);
}

int main(int argc, const char *arg[]) {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  TestMatmulAddBinary<half>(stream);

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
