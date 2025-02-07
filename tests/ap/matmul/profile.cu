#include <cuda.h>
#include <cuda_fp16.h>
#include <functional>
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
    T y;
    half cinn_op_scale_30_0_0 = (x) * 0.1;
    y = cinn_op_scale_30_0_0;
    return y;
  }
};

}

#include "cutlass_matmul.cuh"
#include "profile.h"

int ProfileBestConfig(std::function<void(const ap::GemmEpilogueParams&)>& func, const ap::GemmEpilogueParams &params) {
  std::cout << "Tunning for problem: [" << params.m << ", " << params.n
            << ", " << params.k << "]" << std::endl;

  constexpr int kWarmupIters = 10;
  constexpr int kRepeatIters = 1000;

  for (int i = 0; i < kWarmupIters; i++) {
    func(params);
  }
  if (params.stream) {
    CHECK_CUDA(cudaStreamSynchronize(params.stream));
  }

  GpuTimer gpu_timer(true);
  gpu_timer.Start(params.stream);
  for (int i = 0; i < kRepeatIters; i++) {
    func(params);
  }
  gpu_timer.Stop(params.stream);
  float elapsed_time_ms = gpu_timer.ElapsedTime();
  std::cout << "elapsed_time: " << elapsed_time_ms << " ms" << std::endl;

  return 0;
}

//template <typename TShape = cutlass::gemm::GemmShape<256, 128, 32>,
//          typename WShape = cutlass::gemm::GemmShape<64, 64, 32>,
//          typename IShape = cutlass::gemm::GemmShape<16, 8, 16>,
//          int NumStages = 3>
void RunMatmulAddUnaryKernel(const ap::GemmEpilogueParams& params) {
  using ElementT = cutlass::half_t;
  using ElementComputeT = float;

  ap::UnaryEpilogueFunctor<ElementComputeT>::Arguments unary_args{1.0};
  ap::CutlassMatmulAddUnary<ElementT, ElementComputeT, ap::UnaryEpilogueFunctor, false, false>(params, unary_args);
}

void TuneMatmulAddUnaryKernel(cudaStream_t stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k) {
  ap::GemmEpilogueParams params;

  params.batch_count = batch_count;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output = output;

  params.stream = stream;

  std::function<void(const ap::GemmEpilogueParams&)> matmul_func = RunMatmulAddUnaryKernel;
  ProfileBestConfig(matmul_func, params);
}
