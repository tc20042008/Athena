#include "native_matmul.cuh"

namespace ap {

struct BroadcastConfig {
};

struct InputExtra {
  const void* ptr{nullptr};
  bool need_broadcast{false};
  BroadcastConfig config;
};

template <typename T, int NumIns = 1, int NumOuts = 1>
struct AddFunctor {
  struct Arguments {
    int out_shape_len{0};
    int64_t out_shape[10];
    InputExtra ins[NumIns];
    // void* outs[NumOuts];
  };

  __forceinline__ __host__ __device__
  T Load(const Arguments& args, const MatrixCoord& coord, int idx) const {
    // Specially for the case of out_shape_len = 2
    size_t offset = coord.j * args.out_shape[1] + coord.k;
    return reinterpret_cast<const T*>(args.ins[idx].ptr)[offset];
  }

  __forceinline__ __host__ __device__
  T LoadWithBroadcast(const Arguments& args, const MatrixCoord& coord, int idx) const {
    // Specially for the bias case
    size_t offset = coord.k;
    return reinterpret_cast<const T*>(args.ins[idx].ptr)[offset];
  }

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x, const Arguments& args, const MatrixCoord& coord) const {
    T y = args.ins[0].need_broadcast ? LoadWithBroadcast(args, coord, 0) : Load(args, coord, 0);
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
  epilogue_args.out_shape_len = 2;
  epilogue_args.out_shape[0] = m;
  epilogue_args.out_shape[1] = n;

  epilogue_args.ins[0].ptr = bias;
  epilogue_args.ins[0].need_broadcast = true;

  ap::NativeMatmulAdd<float, ap::AddFunctor<float, 1, 1>>(params, epilogue_args);
}

}
