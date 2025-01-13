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

// template <typename T, int NumIns, int NumOuts>
// struct DoubleAddFunctor {
//   struct Arguments {
//     const void* ins[NumIns];
//     void* outs[NumOuts];
//   };
// 
//   __forceinline__ __host__ __device__
//   T[NumOuts] operator()(T x, T y, T z, Arguments args) const {
//     T[NumOuts] out;
//     out[0] = x + y;
//     out[1] = out[0] + z;
//     return out;
//   }
// };

}
