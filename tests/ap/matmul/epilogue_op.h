#pragma once

namespace ap {

template <typename T>
struct ReluFunctor {
  __forceinline__ __host__ __device__
  T operator()(T x) const {
    return x < static_cast<T>(0) ? static_cast<T>(0) : x;
  }
};

template <typename T>
struct ScaleFunctor {
  struct Arguments {
    using scale_type = T;
    T scale = static_cast<T>(1);
  };

  __forceinline__ __host__ __device__
  T operator()(T x, Arguments args) const {
    return x * args.scale;
  }
};

}
