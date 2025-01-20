#pragma once

namespace ap {

template <typename T>
struct ReluFunctor {
  struct Arguments {};

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return x < static_cast<T>(0) ? static_cast<T>(0) : x;
  }
};

template <typename T>
struct IdentityFunctor {
  struct Arguments {};

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return x;
  }
};

template <typename T>
struct ScaleFunctor {
  struct Arguments {
    float scale = static_cast<float>(1);
  };

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return x * args.scale;
  }
};

struct GemmCoord3d {
  size_t batch_id;
  size_t row_id;
  size_t col_id;
};

template <typename T, int NumIns = 1, int NumOuts = 1>
struct AddFunctor {
  struct Arguments {
    const void* ins[NumIns] = {nullptr};
    // void* outs[NumOuts];
  };

  __forceinline__ __host__ __device__
  T Load(const void* ptr, const GemmCoord3d& coord) const {
    size_t offset = coord.col_id;
    return reinterpret_cast<const T*>(ptr)[offset];
  }

  __forceinline__ __host__ __device__
  T operator()(T x, const GemmCoord3d& coord, const Arguments& args) const {
    T y = Load(args.ins[0], coord);
    return x + y;
  }
};

}
