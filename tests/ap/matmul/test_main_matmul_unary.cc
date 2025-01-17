#include <iostream>

#include "profile.h"
#include "test_util.h"

#ifdef USE_AP_GENERATED_KERNEL
#include "matmul_add_unary_kernel.h"
#else
#include "kernel.h"
#endif


template <typename T>
void TestMatmulAddUnary(cudaStream_t stream, bool add_bias) {
  int batch_count = 4;
  int m = 65536;
  int n = 32;
  int k = 128;

  bool transpose_b = false;

  T* input = AllocateAndInit<T>(stream, batch_count * m * k, false, 1.);
  T* weight = AllocateAndInit<T>(stream, k * n, false, 1.);

  T* bias = nullptr;
  if (add_bias) {
    std::vector<float> bias_ref;
    bias_ref.resize(n);
    for (size_t i = 0; i < bias_ref.size(); ++i) {
      bias_ref[i] = static_cast<float>(1000 * (i % 10));
    }
    bias = AllocateAndInit<T>(stream, n, false, 0., bias_ref);
  }

  T* output = AllocateAndInit<T>(stream, batch_count * m * n, false, 0.);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T) * batch_count * m * n, stream));

#ifdef USE_AP_GENERATED_KERNEL
  KERNEL_PROFILE(MatmulAddUnaryKernel(&stream, input, weight, output, batch_count, m, n, k));
#else
  KERNEL_PROFILE(MatmulAddUnaryKernel(&stream, input, weight, bias, output, batch_count, m, n, k, transpose_b));
#endif

  Print<T>(stream, reinterpret_cast<T*>(output), batch_count, m, n);

  cudaFree(input);
  cudaFree(weight);
  if (add_bias) {
    cudaFree(bias);
  }
  cudaFree(output);
}

int main(int argc, const char *arg[]) {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  TestMatmulAddUnary<half>(stream, false);

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
