// auto generated

#include "epilogue_op.h"

// template <typename T>
// struct EpilogueArguments {
//   typename ap::ScaleFunctor<T>::Arguments scale_args;
// };
 
template <typename T>
struct EpilogueFunctor {
  using Arguments = typename ap::ScaleFunctor<T>::Arguments;

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return ap::ScaleFunctor<T>()(x, args);
  }
};

#include "cutlass_matmul.cuh"

extern "C" {

void MatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b) {
  GemmEpilogueParams params;

  params.batch_count = batch_count;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = stream;

  ap::ScaleFunctor<float>::Arguments unary_args{1.0};
  if (transpose_b) {
    CutlassMatmulAddUnary<cutlass::half_t, float, EpilogueFunctor, false, true>(params, unary_args);
  } else {
    CutlassMatmulAddUnary<cutlass::half_t, float, EpilogueFunctor, false, false>(params, unary_args);
  }
}

void MatmulAddBinaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* broadcast, void* broadcast_out, void* output, int m, int n, int k, bool need_broadcast) {
  GemmBroadcastEpilogueParams params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = stream;

  params.need_broadcast = need_broadcast;
  params.broadcast = broadcast;
  params.broadcast_out = broadcast_out;

  CutlassMatmulAddBinary<cutlass::half_t, float>(params);
}

}
