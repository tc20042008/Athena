#include "epilogue_op.h"

namespace ap {

// template <typename T>
// struct EpilogueArguments {
//   typename ap::ScaleFunctor<T>::Arguments scale_args;
// };

template <typename T>
struct UnaryEpilogueFunctor {
  using Arguments = typename ScaleFunctor<T>::Arguments;

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return ScaleFunctor<T>()(x, args);
  }
};

}

#include "cutlass_matmul.cuh"

extern "C" {

void MatmulAddKernel(cudaStream_t* stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b) {
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

  // std::cout << "-- [MatmulAddKernel] m: " << m << ", n: " << n << ", k: " << k << std::endl;
  // std::cout << "-- [MatmulAddKernel] input: " << input << std::endl;
  // std::cout << "-- [MatmulAddKernel] weight: " << weight << std::endl;
  // std::cout << "-- [MatmulAddKernel] bias: " << bias << std::endl;
  // std::cout << "-- [MatmulAddKernel] output: " << output << std::endl;
  // std::cout << "-- [MatmulAddKernel] stream: " << stream << std::endl;

#if USE_FLOAT16
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  if (transpose_b) {
    ap::CutlassMatmulAdd<ElementT, ElementComputeT, false, true>(params);
  } else {
    ap::CutlassMatmulAdd<ElementT, ElementComputeT, false, false>(params);
  }
}

void MatmulAddUnaryKernel(cudaStream_t* stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b) {
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

#if DEBUG
  std::cout << "-- [MatmulAddUnaryKernel] m: " << m << ", n: " << n << ", k: " << k << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] input: " << input << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] weight: " << weight << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] bias: " << bias << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] output: " << output << std::endl;
  std::cout << "-- [MatmulAddUnaryKernel] stream: " << stream << std::endl;
#endif

#if USE_FLOAT16
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;
#else
  using ElementT = float;
  using ElementComputeT = float;
#endif

  typename ap::UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args{0.1};
  if (transpose_b) {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT, ap::UnaryEpilogueFunctor, false, true>(params, unary_args);
  } else {
    ap::CutlassMatmulAddUnary<ElementT, ElementComputeT, ap::UnaryEpilogueFunctor, false, false>(params, unary_args);
  }
}

void MatmulAddBinaryKernel(cudaStream_t* stream, const void* input, const void* weight, const void* bias, void* broadcast, void* broadcast_out, void* output, int m, int n, int k, bool need_broadcast) {
  ap::GemmBroadcastEpilogueParams params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = *stream;

  params.need_broadcast = need_broadcast;
  params.broadcast = broadcast;
  params.broadcast_out = broadcast_out;

  ap::CutlassMatmulAddBinary<cutlass::half_t, float>(params);
}

}
