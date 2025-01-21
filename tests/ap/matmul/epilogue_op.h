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

}
