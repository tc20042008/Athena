#include "native_matmul.cuh"

namespace ap {

template <typename T, int NumIns = 1, int NumOuts = 1>
struct AddFunctor {
  struct Arguments {
    const void* ins[NumIns] = {nullptr};
    // void* outs[NumOuts];
  };

  __forceinline__ __host__ __device__
  T Load(const void* ptr, const GemmCoord3d& coord) const {
    size_t offset = coord.k;
    return reinterpret_cast<const T*>(ptr)[offset];
  }

  __forceinline__ __host__ __device__
  T operator()(T x, const GemmCoord3d& coord, const Arguments& args) const {
    T y = Load(args.ins[0], coord);
    return x + y;
  }
};

}

extern "C" {

void NativeMatmulAddKernel(cudaStream_t* stream, const void* input, const void* weight, const void* bias, void* output, int batch_count, int m, int n, int k, bool transpose_b) {
  ap::GemmEpilogueParams params;

  params.batch_count = batch_count;
  params.m = m;
  params.n = n;
  params.k = k;

  params.input = input;
  params.weight = weight;
  params.bias = nullptr;
  params.output = output;

  params.stream = *stream;

  typename ap::AddFunctor<float, 1, 1>::Arguments epilogue_args;
  epilogue_args.ins[0] = bias;
  ap::NativeMatmulAdd<float, ap::AddFunctor<float, 1, 1>>(params, epilogue_args);
}

}
