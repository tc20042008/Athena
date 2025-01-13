// auto generated

#include "epilogue_op.h"

// template <typename T>
// struct EpilogueArguments {
//   typename ap::ScaleFunctor<T>::Arguments scale_args;
// };
// 
// template <typename T>
// struct EpilogueFunctor {
//   using Arguments = typename ap::ScaleFunctor<T>::Arguments;
// 
//   __forceinline__ __host__ __device__
//   T operator()(T x, Arguments args) const {
//     return ap::ScaleFunctor<T>()(x, args);
//   }
// };

#include "cutlass_matmul.cuh"

extern "C" {

void MatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int m, int n, int k) {
  GemmEpilogueParams params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = stream;

  ap::ScaleFunctor<float>::Arguments unary_args{0.1};
  CutlassMatmulAddUnary<cutlass::half_t, float, ap::ScaleFunctor>(params, unary_args);

  //EpilogueArguments<float> unary_args{0.1};
  //CutlassMatmulAddUnary<cutlass::half_t, EpilogueFunctor>(params, unary_args);
}

void MatmulAddBinaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* broadcast, void* broadcast_out, void* output, int m, int n, int k) {
  GemmBroadcastEpilogueParams params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.broadcast = broadcast;
  params.broadcast_out = broadcast_out;

  params.stream = stream;

  CutlassMatmulAddBinary<cutlass::half_t, float>(params);
}

}
