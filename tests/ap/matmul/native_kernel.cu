#include "native_matmul.cuh"

extern "C" {

void NativeMatmulAddKernel(cudaStream_t* stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b) {
  ap::GemmEpilogueParams params;

  params.batch_count = batch_count;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = *stream;
  native::MatmulAdd<float>(params);
}

}
